/*!
 * \file eigen.hpp
 * \brief The eigenvalues and eigenvectors of a general square matrix.
 *
 * \ref numerical::eigen_decomposition follows LAPACK's `dgeev` and `zgeev`: the matrix is balanced (permuted to isolate
 * the eigenvalues that need no iteration, then scaled by powers of two so that its rows and columns have comparable
 * norms), reduced to upper Hessenberg form by Householder reflections, and brought to Schur form by the shifted QR
 * algorithm: Francis's implicit double shift for a real matrix (`dlahqr`), whose Schur form is quasi-triangular with
 * a 2 × 2 block for each pair of complex conjugate eigenvalues, and a single Wilkinson shift for a complex one
 * (`zlahqr`). The eigenvectors are those of the Schur form, by back-substitution (`dtrevc`, `ztrevc`), carried back
 * through the reductions; each has unit Euclidean norm and its component of largest modulus real, as LAPACK's.
 *
 * The eigenvalues come in the order the Schur form holds them, which is LAPACK's: no order is promised. A real
 * matrix's complex eigenvalues come in conjugate pairs, the one of positive imaginary part first, and their vectors
 * are conjugate to the bit.
 */
#ifndef NUMERICAL_EIGEN_HPP
#define NUMERICAL_EIGEN_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "decomposition.hpp"

namespace numerical {

  namespace detail {

    //! \brief The relative spacing LAPACK's `dlamch('P')` names: the roundoff a QR sweep judges a subdiagonal by.
    inline constexpr double schur_ulp {std::numeric_limits<double>::epsilon()};

    //! \brief The iterations one eigenvalue (or pair) may take per unit of the active order, as `dlahqr`'s 30.
    inline constexpr std::size_t schur_steps_per_order {30};

    //! \brief The largest power of two a balancing scales a row or a column by, either way: the matrix enters at a
    //!        largest magnitude near 1, so no entry overflows, and the scaling stops where LAPACK's would be one
    //!        factor among dozens.
    inline constexpr double balance_limit {0x1p256};

    //! \brief The modulus an eigenvector's components are kept under during the back-substitution: the sums stay
    //!        finite for any order a matrix can have, and a component that would grow past it scales the vector down.
    inline constexpr double back_substitution_limit {0x1p400};

    //! \brief `|x|` of a real `x`: what \ref modulus_1 of a complex one stands for.
    [[nodiscard]] inline double modulus_1(double x)
    {
      return std::abs(x);
    }

    //! \brief `|re x| + |im x|`, the cheap modulus LAPACK's complex routines judge with.
    [[nodiscard]] inline double modulus_1(const std::complex<double>& x)
    {
      return std::abs(x.real()) + std::abs(x.imag());
    }

    //! \brief How a matrix was balanced: \ref swaps, in the order they were applied, then the scaling of the active
    //!        block `low … high − 1` by \ref scale.
    struct balancing
    {
      std::size_t                                      low  {0}; //!< The active block's first row and column.
      std::size_t                                      high {0}; //!< One past its last.
      std::vector<std::pair<std::size_t, std::size_t>> swaps;    //!< Rows and columns exchanged, in order.
      buffer<double>                                   scale;    //!< The diagonal `D` of `D⁻¹ A D` (1 outside).
    };

    //! \brief Exchanges rows \p i and \p j, then columns \p i and \p j, of the row-major `n × n` \p a: a similarity.
    template <typename T>
    void exchange(buffer<T>&  a,
                  std::size_t n,
                  std::size_t i,
                  std::size_t j)
    {
      for (std::size_t c {0}; c < n; ++c) {
        std::swap(a[(i * n) + c], a[(j * n) + c]);
      }
      for (std::size_t r {0}; r < n; ++r) {
        std::swap(a[(r * n) + i], a[(r * n) + j]);
      }
    }

    /*!
     * \brief Balances the row-major `n × n` \p a in place, as LAPACK's `dgebal('B')`: rows with no off-diagonal
     *        entry in the active block go to its bottom and columns with none to its top, each isolating an
     *        eigenvalue on the diagonal; then each row and column of the remaining block is scaled by a power of two
     *        until their 2-norms are within a factor of 2, which leaves every entry exact.
     */
    template <typename T>
    [[nodiscard]] balancing balance(buffer<T>&  a,
                                    std::size_t n)
    {
      balancing out;
      out.scale = buffer<double>(n, 1.0);
      out.high  = n;
      for (bool moved {true}; moved;) {
        moved = false;
        for (std::size_t i {out.high}; i-- > out.low;) {
          bool alone {true};
          for (std::size_t j {out.low}; j < out.high && alone; ++j) {
            alone = j == i || a[(i * n) + j] == T {};
          }
          if (alone) {
            exchange(a, n, i, out.high - 1);
            out.swaps.emplace_back(i, out.high - 1);
            --out.high;
            moved = true;
            break;
          }
        }
      }
      for (bool moved {true}; moved;) {
        moved = false;
        for (std::size_t j {out.low}; j < out.high; ++j) {
          bool alone {true};
          for (std::size_t i {out.low}; i < out.high && alone; ++i) {
            alone = i == j || a[(i * n) + j] == T {};
          }
          if (alone) {
            exchange(a, n, j, out.low);
            out.swaps.emplace_back(j, out.low);
            ++out.low;
            moved = true;
            break;
          }
        }
      }
      for (bool unbalanced {true}; unbalanced;) {
        unbalanced = false;
        for (std::size_t i {out.low}; i < out.high; ++i) {
          double column {0.0};
          double row    {0.0};
          for (std::size_t k {out.low}; k < out.high; ++k) {
            column = std::hypot(column, std::abs(a[(k * n) + i]));
            row    = std::hypot(row, std::abs(a[(i * n) + k]));
          }
          // Both are positive: the exchanges left every row and column of the block an off-diagonal entry in it.
          const double before {column + row};
          double       f      {1.0};
          while (2.0 * column < row && out.scale[i] * f < balance_limit) {
            f      *= 2.0;
            column *= 2.0;
            row    /= 2.0;
          }
          while (column >= 2.0 * row && out.scale[i] * f > 1.0 / balance_limit) {
            f      /= 2.0;
            column /= 2.0;
            row    *= 2.0;
          }
          if (column + row >= 0.95 * before) {
            continue;
          }
          out.scale[i] *= f;
          unbalanced    = true;
          for (std::size_t k {out.low}; k < n; ++k) {
            a[(i * n) + k] /= from_double<T>(f);
          }
          for (std::size_t k {0}; k < out.high; ++k) {
            a[(k * n) + i] *= from_double<T>(f);
          }
        }
      }
      return out;
    }

    /*!
     * \brief Reduces the active block `low … high − 1` of the balanced row-major `n × n` \p a to upper Hessenberg form
     *        by Householder reflections, as LAPACK's `dgehd2`: `A = Q H Qᴴ`, the entries below the subdiagonal set
     *        to zero.
     * \param[in,out] a    The matrix; on return `H`.
     * \param[in]     n    Its order.
     * \param[in]     low  The active block's first row and column.
     * \param[in]     high One past its last.
     * \param[in]     form Whether to form `Q`.
     * \return `Qᵀ`, row-major `n × n` (the identity outside the active block), or empty without \p form: a QR sweep
     *         updates columns of `Q`, which are then contiguous.
     */
    template <typename T>
    [[nodiscard]] buffer<T> hessenberg(buffer<T>&  a,
                                       std::size_t n,
                                       std::size_t low,
                                       std::size_t high,
                                       bool        form)
    {
      std::vector<buffer<T>> reflectors(n); // reflectors[k] acts on rows (and columns) k + 1 … high − 1
      for (std::size_t k {low}; k + 2 < high; ++k) {
        buffer<T> v(high - k - 1, T {});
        for (std::size_t i {k + 1}; i < high; ++i) {
          v[i - k - 1] = a[(i * n) + k];
        }
        if (scaled_norm(v.data() + 1, v.size() - 1) == 0.0) {
          continue;
        }
        (void)rescale_tiny(v); // the same reflector at any scale
        const double norm {scaled_norm(v)};
        v[0] += phase_of(v[0]) * from_double<T>(norm);
        const T length    {from_double<T>(scaled_norm(v))};
        for (T& element : v) {
          element /= length;
        }
        buffer<T> w(n - k, T {}); // the rows times (I − 2 v vᴴ) from the left: w = vᴴ A over columns k …
        for (std::size_t i {0}; i < v.size(); ++i) {
          const T  weight {conj_of(v[i])};
          const T* row    {a.data() + ((k + 1 + i) * n) + k};
          for (std::size_t c {0}; c < w.size(); ++c) {
            w[c] += weight * row[c];
          }
        }
        for (std::size_t i {0}; i < v.size(); ++i) {
          const T twice {static_cast<T>(2) * v[i]};
          T*      row   {a.data() + ((k + 1 + i) * n) + k};
          for (std::size_t c {0}; c < w.size(); ++c) {
            row[c] -= twice * w[c];
          }
        }
        for (std::size_t r {0}; r < high; ++r) { // A (I − 2 v vᴴ) on columns k + 1 …; rows below are zero there
          T* row {a.data() + (r * n) + k + 1};
          T  s   {};
          for (std::size_t j {0}; j < v.size(); ++j) {
            s += row[j] * v[j];
          }
          s *= static_cast<T>(2);
          for (std::size_t j {0}; j < v.size(); ++j) {
            row[j] -= s * conj_of(v[j]);
          }
        }
        for (std::size_t i {k + 2}; i < high; ++i) {
          a[(i * n) + k] = T {};
        }
        reflectors[k] = std::move(v);
      }
      if (!form) {
        return {};
      }
      buffer<T> q(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        q[(i * n) + i] = T {1};
      }
      for (std::size_t k {n}; k-- > 0;) { // Q = P_low … P_high−3, accumulated backward: each acts on rows k + 1 …
        const buffer<T>& v {reflectors[k]};
        if (v.empty()) {
          continue;
        }
        buffer<T> w(v.size(), T {});
        for (std::size_t i {0}; i < v.size(); ++i) {
          const T  weight {conj_of(v[i])};
          const T* row    {q.data() + ((k + 1 + i) * n) + k + 1};
          for (std::size_t c {0}; c < w.size(); ++c) {
            w[c] += weight * row[c];
          }
        }
        for (std::size_t i {0}; i < v.size(); ++i) {
          const T twice {static_cast<T>(2) * v[i]};
          T*      row   {q.data() + ((k + 1 + i) * n) + k + 1};
          for (std::size_t c {0}; c < w.size(); ++c) {
            row[c] -= twice * w[c];
          }
        }
      }
      buffer<T> qt(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          qt[(j * n) + i] = q[(i * n) + j];
        }
      }
      return qt;
    }

    /*!
     * \brief The Householder reflection `I − τ v vᵀ`, `v = (1, x…)`, that takes `(α, x…)` to `(β, 0…)`, as LAPACK's
     *        `dlarfg` for two or three entries: on return `v[0] = β` and `v[1…]` holds `x`'s part of `v`.
     * \return `τ`, 0 where `x` is zero (the identity).
     */
    [[nodiscard]] inline double small_reflector(double (&v)[3],
                                                std::size_t count)
    {
      const double rest {count == 3 ? std::hypot(v[1], v[2]) : std::abs(v[1])};
      if (rest == 0.0) {
        return 0.0;
      }
      const double beta  {-std::copysign(std::hypot(v[0], rest), v[0])};
      const double tau   {(beta - v[0]) / beta};
      const double scale {1.0 / (v[0] - beta)};
      for (std::size_t i {1}; i < count; ++i) {
        v[i] *= scale;
      }
      v[0] = beta;
      return tau;
    }

    /*!
     * \brief The Householder reflection `I − τ v vᴴ`, `v = (1, x)`, whose adjoint takes `(α, x)` to `(β, 0)`, `β` real,
     *        as LAPACK's `zlarfg` for two entries: on return `v[0] = β` and `v[1]` holds `x`'s part of `v`.
     * \return `τ`, 0 where `x` is zero and `α` real (the identity).
     */
    [[nodiscard]] inline std::complex<double> complex_reflector(std::complex<double> (&v)[2])
    {
      if (modulus_1(v[1]) == 0.0 && v[0].imag() == 0.0) {
        return {};
      }
      const double               beta {-std::copysign(std::hypot(std::abs(v[0]), std::abs(v[1])), v[0].real())};
      const std::complex<double> tau  {(beta - v[0].real()) / beta, -v[0].imag() / beta};
      v[1] = v[1] / (v[0] - beta);
      v[0] = beta;
      return tau;
    }

    //! \brief The rotation \ref standardize_block found, and the eigenvalues of the block it standardised.
    struct standard_block
    {
      double cs  {1.0};   //!< Its cosine.
      double sn  {0.0};   //!< Its sine.
      double re1 {0.0};   //!< The first eigenvalue's real part.
      double im1 {0.0};   //!< Its imaginary part (positive for a complex pair).
      double re2 {0.0};   //!< The second's real part.
      double im2 {0.0};   //!< Its imaginary part (−\ref im1).
    };

    /*!
     * \brief The Schur factorisation of the real 2 × 2 block `[a b; c d]`, as LAPACK's `dlanv2`:
     *        `[a b; c d] ← [cs −sn; sn cs]ᵀ [a b; c d] [cs −sn; sn cs]`, upper triangular where its eigenvalues are
     *        real, else with equal diagonal entries and `b c < 0`.
     */
    [[nodiscard]] inline standard_block standardize_block(double& a,
                                                          double& b,
                                                          double& c,
                                                          double& d)
    {
      standard_block out;
      if (b == 0.0 && c != 0.0) { // lower triangular: exchange the rows and the columns
        out.cs = 0.0;
        out.sn = 1.0;
        std::swap(a, d);
        b = -c;
        c = 0.0;
      }
      else if (c != 0.0 && (a - d != 0.0 || std::signbit(b) == std::signbit(c))) { // not yet triangular nor standard
        const double temp  {a - d};
        double       p     {0.5 * temp};
        const double bcmax {std::max(std::abs(b), std::abs(c))};
        const double bcmis {std::min(std::abs(b), std::abs(c)) * std::copysign(1.0, b) * std::copysign(1.0, c)};
        const double scale {std::max(std::abs(p), bcmax)};
        double       z     {((p / scale) * p) + ((bcmax / scale) * bcmis)};
        if (z >= 4.0 * schur_ulp) { // real eigenvalues: the triangular form
          z      = p + std::copysign(std::sqrt(scale) * std::sqrt(z), p);
          a      = d + z;
          d     -= (bcmax / z) * bcmis;
          const double tau {std::hypot(c, z)};
          out.cs = z / tau;
          out.sn = c / tau;
          b     -= c;
          c      = 0.0;
        }
        else { // complex or nearly equal real eigenvalues: equal diagonal entries first
          const double sigma {b + c};
          const double tau   {std::hypot(sigma, temp)};
          out.cs = std::sqrt(0.5 * (1.0 + (std::abs(sigma) / tau)));
          out.sn = -(0.5 * (temp / tau) / out.cs) * std::copysign(1.0, sigma);
          const double aa {(a * out.cs) + (b * out.sn)};
          const double bb {(-a * out.sn) + (b * out.cs)};
          const double cc {(c * out.cs) + (d * out.sn)};
          const double dd {(-c * out.sn) + (d * out.cs)};
          a = (aa * out.cs) + (cc * out.sn);
          b = (bb * out.cs) + (dd * out.sn);
          c = (-aa * out.sn) + (cc * out.cs);
          d = (-bb * out.sn) + (dd * out.cs);
          p = 0.5 * (a + d);
          a = p;
          d = p;
          if (c != 0.0 && b != 0.0 && std::signbit(b) == std::signbit(c)) { // real after all: triangular
            const double sab     {std::sqrt(std::abs(b))};
            const double sac     {std::sqrt(std::abs(c))};
            const double root    {std::copysign(sab * sac, c)};
            const double inverse {1.0 / std::sqrt(std::abs(b + c))};
            a  = p + root;
            d  = p - root;
            b -= c;
            c  = 0.0;
            const double cs1 {sab * inverse};
            const double sn1 {sac * inverse};
            const double cs  {(out.cs * cs1) - (out.sn * sn1)};
            out.sn = (out.cs * sn1) + (out.sn * cs1);
            out.cs = cs;
          }
          else if (c != 0.0 && b == 0.0) {
            b = -c;
            c = 0.0;
            const double cs {out.cs};
            out.cs = -out.sn;
            out.sn = cs;
          }
        }
      }
      out.re1 = a;
      out.re2 = d;
      if (c != 0.0) {
        out.im1 = std::sqrt(std::abs(b)) * std::sqrt(std::abs(c));
        out.im2 = -out.im1;
      }
      return out;
    }

    //! \brief `(x, y) ← (cs x + sn y, cs y − sn x)` over \p count pairs \p stride apart: a plane rotation.
    template <typename T>
    void rotate(T*          x,
                T*          y,
                std::size_t count,
                std::size_t stride,
                double      cs,
                double      sn)
    {
      for (std::size_t i {0}; i < count; ++i) {
        const T xi {x[i * stride]};
        const T yi {y[i * stride]};
        x[i * stride] = (cs * xi) + (sn * yi);
        y[i * stride] = (cs * yi) - (sn * xi);
      }
    }

    //! \brief Where a QR sweep applies its reflections: the rows and columns it updates, and `Z`.
    template <typename T>
    struct schur_target
    {
      buffer<T>*  h    {nullptr}; //!< The Hessenberg matrix, row-major.
      std::size_t n    {0};       //!< Its order.
      std::size_t low  {0};       //!< The active block's first row.
      std::size_t high {0};       //!< One past its last.
      bool        full {false};   //!< Whether the whole Schur form is wanted (else the active block alone).
      buffer<T>*  zt   {nullptr}; //!< The Schur vectors to accumulate, transposed (row-major `Zᵀ`), or none.
    };

    /*!
     * \brief The real Schur form of the upper Hessenberg \p t's active block, as LAPACK's `dlahqr`: Francis's implicit
     *        double-shift QR, with the deflation criterion of Ahues and Tisseur and exceptional shifts after 10 and
     *        20 iterations without deflation. The eigenvalues are left in \p re and \p im, a complex pair's in two
     *        consecutive places, the positive imaginary part first.
     * \param[in]     t     The Hessenberg matrix, its active block, and the Schur vectors to update.
     * \param[in,out] re    The eigenvalues' real parts (the isolated ones already in place).
     * \param[out]    im    Their imaginary parts.
     * \param[in]     steps The iterations allowed between two deflations.
     * \throws convergence_error If an eigenvalue has not converged within \p steps.
     */
    inline void real_schur(const schur_target<double>& t,
                           buffer<double>&             re,
                           buffer<double>&             im,
                           std::size_t                 steps)
    {
      buffer<double>&   h {*t.h};
      const std::size_t n {t.n};
      const auto        at {[&h, n](std::size_t i, std::size_t j) -> double& { return h[(i * n) + j]; }};
      const double      smallest        {std::numeric_limits<double>::min() * (static_cast<double>(t.high - t.low) / schur_ulp)};
      std::size_t       since_deflation {0};
      for (std::size_t end {t.high}; end > t.low;) {
        const std::size_t i         {end - 1};
        std::size_t       l         {t.low};
        bool              converged {false};
        for (std::size_t its {0}; its <= steps; ++its) {
          std::size_t k {i};
          for (; k > l; --k) {
            const double sub {std::abs(at(k, k - 1))};
            if (sub <= smallest) {
              break;
            }
            double test {std::abs(at(k - 1, k - 1)) + std::abs(at(k, k))};
            if (test == 0.0) {
              if (k >= t.low + 2) {
                test += std::abs(at(k - 1, k - 2));
              }
              if (k + 1 < t.high) {
                test += std::abs(at(k + 1, k));
              }
            }
            if (sub <= schur_ulp * test) {
              const double ab {std::max(sub, std::abs(at(k - 1, k)))};
              const double ba {std::min(sub, std::abs(at(k - 1, k)))};
              const double aa {std::max(std::abs(at(k, k)), std::abs(at(k - 1, k - 1) - at(k, k)))};
              const double bb {std::min(std::abs(at(k, k)), std::abs(at(k - 1, k - 1) - at(k, k)))};
              const double s  {aa + ab};
              if (ba * (ab / s) <= std::max(smallest, schur_ulp * (bb * (aa / s)))) {
                break;
              }
            }
          }
          l = k;
          if (l > t.low) {
            at(l, l - 1) = 0.0;
          }
          if (l + 1 >= i) {
            converged = true;
            break;
          }
          ++since_deflation;
          const std::size_t first {t.full ? 0 : l};     // the columns the right reflections update start here
          const std::size_t last  {t.full ? n : i + 1}; // the rows the left reflections update end before here
          double            h11   {};
          double            h12   {};
          double            h21   {};
          double            h22   {};
          if (since_deflation % 20 == 0) {
            const double s {std::abs(at(i, i - 1)) + std::abs(at(i - 1, i - 2))};
            h11 = (0.75 * s) + at(i, i);
            h12 = -0.4375 * s;
            h21 = s;
            h22 = h11;
          }
          else if (since_deflation % 10 == 0) {
            const double s {std::abs(at(l + 1, l)) + std::abs(at(l + 2, l + 1))};
            h11 = (0.75 * s) + at(l, l);
            h12 = -0.4375 * s;
            h21 = s;
            h22 = h11;
          }
          else {
            h11 = at(i - 1, i - 1);
            h21 = at(i, i - 1);
            h12 = at(i - 1, i);
            h22 = at(i, i);
          }
          const double s {std::abs(h11) + std::abs(h12) + std::abs(h21) + std::abs(h22)}; // > 0: h21 is not negligible
          h11 /= s;
          h21 /= s;
          h12 /= s;
          h22 /= s;
          const double trace {(h11 + h22) / 2.0};
          const double det   {((h11 - trace) * (h22 - trace)) - (h12 * h21)};
          const double root  {std::sqrt(std::abs(det))};
          double       rt1r  {trace * s};
          double       rt2r  {rt1r};
          double       rt1i  {root * s};
          double       rt2i  {-rt1i};
          if (det < 0.0) { // real shifts: the one nearer h22, twice
            rt1r = trace + root;
            rt2r = trace - root;
            rt1r = (std::abs(rt1r - h22) <= std::abs(rt2r - h22) ? rt1r : rt2r) * s;
            rt2r = rt1r;
            rt1i = 0.0;
            rt2i = 0.0;
          }
          std::size_t m    {i - 2};
          double      v[3] {};
          for (;; --m) { // where two consecutive small subdiagonals let the sweep start
            const double h21s  {at(m + 1, m) / (std::abs(at(m, m) - rt2r) + std::abs(rt2i) + std::abs(at(m + 1, m)))};
            const double scale {std::abs(at(m, m) - rt2r) + std::abs(rt2i) + std::abs(at(m + 1, m))};
            v[0] = (h21s * at(m, m + 1)) + ((at(m, m) - rt1r) * ((at(m, m) - rt2r) / scale)) - (rt1i * (rt2i / scale));
            v[1] = h21s * (at(m, m) + at(m + 1, m + 1) - rt1r - rt2r);
            v[2] = h21s * at(m + 2, m + 1);
            const double sum {std::abs(v[0]) + std::abs(v[1]) + std::abs(v[2])};
            v[0] /= sum;
            v[1] /= sum;
            v[2] /= sum;
            if (m == l) {
              break;
            }
            const double above {std::abs(at(m, m - 1)) * (std::abs(v[1]) + std::abs(v[2]))};
            const double here  {std::abs(v[0]) * (std::abs(at(m - 1, m - 1)) + std::abs(at(m, m)) + std::abs(at(m + 1, m + 1)))};
            if (above <= schur_ulp * here) {
              break;
            }
          }
          for (std::size_t k2 {m}; k2 < i; ++k2) { // the bulge chased down the diagonal
            const std::size_t count {std::min<std::size_t>(3, i - k2 + 1)};
            if (k2 > m) {
              v[0] = at(k2, k2 - 1);
              v[1] = at(k2 + 1, k2 - 1);
              v[2] = count == 3 ? at(k2 + 2, k2 - 1) : 0.0;
            }
            const double t1 {small_reflector(v, count)};
            if (k2 > m) {
              at(k2, k2 - 1)     = v[0];
              at(k2 + 1, k2 - 1) = 0.0;
              if (count == 3) {
                at(k2 + 2, k2 - 1) = 0.0;
              }
            }
            else if (m > l) {
              at(k2, k2 - 1) *= 1.0 - t1;
            }
            const double      v2    {v[1]};
            const double      v3    {count == 3 ? v[2] : 0.0};
            const double      t2    {t1 * v2};
            const double      t3    {t1 * v3};
            const std::size_t third {count == 3 ? std::size_t {2} : std::size_t {1}}; // two rows: the third is the second, t3 = 0 leaves it
            double*           a1    {&at(k2, 0)};
            double*           a2    {a1 + n};
            double*           a3    {a1 + (third * n)};
            for (std::size_t j {k2}; j < last; ++j) {
              const double sum {a1[j] + (v2 * a2[j]) + (v3 * a3[j])};
              a1[j] -= sum * t1;
              a2[j] -= sum * t2;
              a3[j] -= sum * t3;
            }
            for (std::size_t j {first}; j <= std::min(k2 + 3, i); ++j) {
              double*      row {&at(j, k2)};
              const double sum {row[0] + (v2 * row[1]) + (v3 * row[third])};
              row[0]     -= sum * t1;
              row[1]     -= sum * t2;
              row[third] -= sum * t3;
            }
            if (t.zt != nullptr) { // Z's columns k2 …, rows of Zᵀ
              buffer<double>& zt {*t.zt};
              double*         r1 {zt.data() + (k2 * n)};
              double*         r2 {r1 + n};
              double*         r3 {r1 + (third * n)};
              for (std::size_t j {t.low}; j < t.high; ++j) {
                const double sum {r1[j] + (v2 * r2[j]) + (v3 * r3[j])};
                r1[j] -= sum * t1;
                r2[j] -= sum * t2;
                r3[j] -= sum * t3; // r3 is r2 and t3 0 for two rows
              }
            }
          }
        }
        if (!converged) {
          throw convergence_error("eigendecomposition: the QR iterations did not converge");
        }
        if (l == i) {
          re[i] = at(i, i);
          im[i] = 0.0;
        }
        else { // a 2 × 2 block: its standard form, the rotation carried to the rest of the Schur form and Z
          const standard_block block {standardize_block(at(i - 1, i - 1), at(i - 1, i), at(i, i - 1), at(i, i))};
          re[i - 1] = block.re1;
          im[i - 1] = block.im1;
          re[i]     = block.re2;
          im[i]     = block.im2;
          if (t.full) {
            rotate(&at(i - 1, i + 1 < n ? i + 1 : i), &at(i, i + 1 < n ? i + 1 : i), n - i - 1, 1, block.cs, block.sn);
            rotate(&at(0, i - 1), &at(0, i), i - 1, n, block.cs, block.sn);
          }
          if (t.zt != nullptr) {
            buffer<double>& zt {*t.zt};
            rotate(zt.data() + ((i - 1) * n) + t.low, zt.data() + (i * n) + t.low, t.high - t.low, 1, block.cs, block.sn);
          }
        }
        since_deflation = 0;
        end             = l;
      }
    }

    /*!
     * \brief The complex Schur form of the upper Hessenberg \p t's active block, as LAPACK's `zlahqr`: the shifted QR
     *        algorithm with Wilkinson's shift, the deflation criterion of Ahues and Tisseur, and exceptional shifts
     *        after 10 and 20 iterations without deflation. The eigenvalues are left in \p values.
     * \param[in]     t      The Hessenberg matrix, its active block, and the Schur vectors to update.
     * \param[in,out] values The eigenvalues (the isolated ones already in place).
     * \param[in]     steps  The iterations allowed between two deflations.
     * \throws convergence_error If an eigenvalue has not converged within \p steps.
     */
    inline void complex_schur(const schur_target<std::complex<double>>& t,
                              buffer<std::complex<double>>&             values,
                              std::size_t                               steps)
    {
      using C = std::complex<double>;
      buffer<C>&        h {*t.h};
      const std::size_t n {t.n};
      const auto        at {[&h, n](std::size_t i, std::size_t j) -> C& { return h[(i * n) + j]; }};
      const double      smallest        {std::numeric_limits<double>::min() * (static_cast<double>(t.high - t.low) / schur_ulp)};
      std::size_t       since_deflation {0};
      for (std::size_t end {t.high}; end > t.low;) {
        const std::size_t i         {end - 1};
        std::size_t       l         {t.low};
        bool              converged {false};
        for (std::size_t its {0}; its <= steps; ++its) {
          std::size_t k {i};
          for (; k > l; --k) {
            const double sub {modulus_1(at(k, k - 1))};
            if (sub <= smallest) {
              break;
            }
            double test {modulus_1(at(k - 1, k - 1)) + modulus_1(at(k, k))};
            if (test == 0.0) {
              if (k >= t.low + 2) {
                test += modulus_1(at(k - 1, k - 2));
              }
              if (k + 1 < t.high) {
                test += modulus_1(at(k + 1, k));
              }
            }
            if (sub <= schur_ulp * test) {
              const double ab {std::max(sub, modulus_1(at(k - 1, k)))};
              const double ba {std::min(sub, modulus_1(at(k - 1, k)))};
              const double aa {std::max(modulus_1(at(k, k)), modulus_1(at(k - 1, k - 1) - at(k, k)))};
              const double bb {std::min(modulus_1(at(k, k)), modulus_1(at(k - 1, k - 1) - at(k, k)))};
              const double s  {aa + ab};
              if (ba * (ab / s) <= std::max(smallest, schur_ulp * (bb * (aa / s)))) {
                break;
              }
            }
          }
          l = k;
          if (l > t.low) {
            at(l, l - 1) = C {};
          }
          if (l >= i) {
            converged = true;
            break;
          }
          ++since_deflation;
          const std::size_t first {t.full ? 0 : l};
          const std::size_t last  {t.full ? n : i + 1};
          C                 shift {};
          if (since_deflation % 20 == 0) {
            shift = (0.75 * modulus_1(at(i, i - 1))) + at(i, i);
          }
          else if (since_deflation % 10 == 0) {
            shift = (0.75 * modulus_1(at(l + 1, l))) + at(l, l);
          }
          else { // Wilkinson's: the eigenvalue of the trailing 2 × 2 nearer its last diagonal entry
            shift = at(i, i);
            const C u {std::sqrt(at(i - 1, i)) * std::sqrt(at(i, i - 1))};
            if (modulus_1(u) != 0.0) {
              const C      x {0.5 * (at(i - 1, i - 1) - shift)};
              const double sx {modulus_1(x)};
              const double s {std::max(modulus_1(u), sx)};
              C            y {s * std::sqrt(((x / s) * (x / s)) + ((u / s) * (u / s)))};
              if (sx > 0.0 && ((x / sx).real() * y.real()) + ((x / sx).imag() * y.imag()) < 0.0) {
                y = -y;
              }
              shift -= u * (u / (x + y));
            }
          }
          std::size_t m    {i - 1};
          C           v[2] {};
          for (;; --m) {
            const C      h11 {at(m, m)};
            const C      h22 {at(m + 1, m + 1)};
            const C      h21 {at(m + 1, m)};
            const double s   {modulus_1(h11 - shift) + modulus_1(h21)};
            v[0] = (h11 - shift) / s;
            v[1] = h21 / s;
            if (m == l) {
              break;
            }
            if (modulus_1(at(m, m - 1)) * modulus_1(v[1]) <= schur_ulp * (modulus_1(v[0]) * (modulus_1(h11) + modulus_1(h22)))) {
              break;
            }
          }
          for (std::size_t k2 {m}; k2 < i; ++k2) {
            if (k2 > m) {
              v[0] = at(k2, k2 - 1);
              v[1] = at(k2 + 1, k2 - 1);
            }
            const C t1 {complex_reflector(v)};
            if (k2 > m) {
              at(k2, k2 - 1)     = v[0];
              at(k2 + 1, k2 - 1) = C {};
            }
            else if (m > l) {
              at(k2, k2 - 1) *= 1.0 - std::conj(t1);
            }
            const C v2 {v[1]};
            for (std::size_t j {k2}; j < last; ++j) {
              const C sum {std::conj(t1) * (at(k2, j) + (std::conj(v2) * at(k2 + 1, j)))};
              at(k2, j)     -= sum;
              at(k2 + 1, j) -= sum * v2;
            }
            for (std::size_t j {first}; j <= std::min(k2 + 2, i); ++j) {
              const C sum {t1 * (at(j, k2) + (v2 * at(j, k2 + 1)))};
              at(j, k2)     -= sum;
              at(j, k2 + 1) -= sum * std::conj(v2);
            }
            if (t.zt != nullptr) {
              C* r1 {t.zt->data() + (k2 * n)};
              C* r2 {r1 + n};
              for (std::size_t j {t.low}; j < t.high; ++j) {
                const C sum {t1 * (r1[j] + (v2 * r2[j]))};
                r1[j] -= sum;
                r2[j] -= sum * std::conj(v2);
              }
            }
          }
        }
        if (!converged) {
          throw convergence_error("eigendecomposition: the QR iterations did not converge");
        }
        values[i]       = at(i, i);
        since_deflation = 0;
        end             = l;
      }
    }

    /*!
     * \brief Solves `[a b; c d] x = r` in place of \p r by Gaussian elimination with complete pivoting, a pivot below
     *        \p floor taken as \p floor, as LAPACK's `dlaln2`; \p r is scaled first, and \return the factor, wherever
     *        `x` would pass \ref back_substitution_limit.
     */
    template <typename S>
    [[nodiscard]] double solve_block(const S (&m)[4],
                                     S (&r)[2],
                                     double floor)
    {
      std::size_t p {0};
      for (std::size_t q {1}; q < 4; ++q) {
        p = modulus_1(m[q]) > modulus_1(m[p]) ? q : p;
      }
      const std::size_t pr  {p / 2};
      const std::size_t pc  {p % 2};
      const S           u11 {modulus_1(m[p]) < floor ? S {floor} : m[p]};
      const S           l21 {m[((1 - pr) * 2) + pc] / u11};
      const S           u12 {m[(pr * 2) + 1 - pc]};
      S                 u22 {m[((1 - pr) * 2) + 1 - pc] - (l21 * u12)};
      if (modulus_1(u22) < floor) {
        u22 = S {floor};
      }
      const double rmax   {std::max(modulus_1(r[0]), modulus_1(r[1]))};
      const double pivot  {std::min(modulus_1(u11), modulus_1(u22))};
      double       factor {1.0};
      if (4.0 * rmax > back_substitution_limit * pivot) {
        factor = back_substitution_limit * pivot / (4.0 * rmax);
        r[0]  *= factor;
        r[1]  *= factor;
      }
      const S rp {r[pr]};
      const S ro {r[1 - pr] - (l21 * rp)};
      r[1 - pc] = ro / u22;
      r[pc]     = (rp - (u12 * r[1 - pc])) / u11;
      return factor;
    }

    //! \brief `x[j] = r / pivot`, \p r and the components \p x `[from…to]` scaled first where `x[j]` would pass
    //!        \ref back_substitution_limit; a pivot below \p floor is taken as \p floor.
    template <typename S, typename P>
    void divide_component(buffer<S>&  x,
                          std::size_t j,
                          S           r,
                          P           pivot,
                          double      floor,
                          std::size_t to)
    {
      const double size    {modulus_1(pivot) < floor ? floor : modulus_1(pivot)};
      const P      divisor {modulus_1(pivot) < floor ? P {floor} : pivot};
      if (modulus_1(r) > back_substitution_limit * size) {
        const double factor {back_substitution_limit * size / (2.0 * modulus_1(r))};
        for (std::size_t l {j + 1}; l <= to; ++l) {
          x[l] *= factor;
        }
        r *= factor;
      }
      x[j] = r / divisor;
    }

    //! \brief `Σ_{l = from…to} t(j, l) x[l]`, a row of the row-major \p t against \p x.
    template <typename T, typename S>
    [[nodiscard]] S row_against(const buffer<T>& t,
                                std::size_t      n,
                                std::size_t      j,
                                const buffer<S>& x,
                                std::size_t      from,
                                std::size_t      to)
    {
      S sum {};
      for (std::size_t l {from}; l <= to; ++l) {
        sum += t[(j * n) + l] * x[l];
      }
      return sum;
    }

    //! \brief Scales \p v to unit Euclidean norm and turns it so that its first component of largest modulus is real:
    //!        positive for a complex matrix (`zgeev`), of its real part's sign for a real one's pair (`dgeev`).
    inline void normalize_vector(std::complex<double>* v,
                                 std::size_t           n,
                                 bool                  signed_real)
    {
      const double norm    {scaled_norm(v, n)};
      std::size_t  largest {0};
      for (std::size_t r {0}; r < n; ++r) {
        v[r]    /= norm;
        largest  = std::norm(v[r]) > std::norm(v[largest]) ? r : largest;
      }
      std::complex<double> turn {std::conj(v[largest]) / std::abs(v[largest])};
      if (signed_real && std::signbit(v[largest].real())) {
        turn = -turn;
      }
      for (std::size_t r {0}; r < n; ++r) {
        v[r] *= turn;
      }
      v[largest] = std::complex<double> {v[largest].real(), 0.0};
    }

    /*!
     * \brief Solves `(T − λ I) x = 0` upward from the component \p k, rows \p k − 1 … 0 of the row-major quasi-triangular
     *        \p t (its 2 × 2 blocks where a subdiagonal entry is nonzero), \p x `[k … top]` given: as LAPACK's `dtrevc`
     *        and `ztrevc`, a pivot below \p floor taken as \p floor, the components scaled down where they would grow
     *        past \ref back_substitution_limit.
     */
    template <typename T, typename S>
    void solve_upward(const buffer<T>& t,
                      std::size_t      n,
                      std::size_t      k,
                      std::size_t      top,
                      S                lambda,
                      double           floor,
                      buffer<S>&       x)
    {
      for (std::size_t j {k}; j > 0;) {
        if constexpr (std::is_floating_point_v<T>) {
          if (j >= 2 && t[((j - 1) * n) + j - 2] != 0.0) { // a 2 × 2 block, rows j − 2 and j − 1
            const S      m[4]   {S {t[((j - 2) * n) + j - 2]} - lambda, S {t[((j - 2) * n) + j - 1]},
                                 S {t[((j - 1) * n) + j - 2]}, S {t[((j - 1) * n) + j - 1]} - lambda};
            S            r[2]   {-row_against(t, n, j - 2, x, j, top), -row_against(t, n, j - 1, x, j, top)};
            const double factor {solve_block(m, r, floor)};
            for (std::size_t l {j}; l <= top; ++l) {
              x[l] *= factor;
            }
            x[j - 2]  = r[0];
            x[j - 1]  = r[1];
            j        -= 2;
            continue;
          }
        }
        divide_component(x, j - 1, -row_against(t, n, j - 1, x, j, top), S {t[((j - 1) * n) + j - 1]} - lambda, floor, top);
        --j;
      }
    }

    /*!
     * \brief The eigenvectors of the complex upper triangular \p t (row-major `n × n`) as the rows of `Xᵀ`: row `k`
     *        is `x` with `(T − t_kk I) x = 0`, `x_k = 1`, zero past `k`, as LAPACK's `ztrevc`.
     */
    [[nodiscard]] inline buffer<std::complex<double>> complex_vectors(const buffer<std::complex<double>>& t,
                                                                      std::size_t                         n)
    {
      using C = std::complex<double>;
      const double smallest    {std::numeric_limits<double>::min() * (static_cast<double>(n) / schur_ulp)};
      buffer<C>    xt(n * n, C {});
      buffer<C>    x(n, C      {});
      for (std::size_t k {0}; k < n; ++k) {
        const C lambda {t[(k * n) + k]};
        x[k] = C {1.0};
        solve_upward(t, n, k, k, lambda, std::max(schur_ulp * modulus_1(lambda), smallest), x);
        std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(k + 1), xt.begin() + static_cast<std::ptrdiff_t>(k * n));
      }
      return xt;
    }

    //! \brief A real matrix's eigenvectors before their back-transformation: `Xᵀ`'s real and imaginary parts.
    struct real_eigenvectors
    {
      buffer<double> re; //!< `Re Xᵀ`, row-major `n × n`.
      buffer<double> im; //!< `Im Xᵀ`, zero where every eigenvalue is real.
    };

    /*!
     * \brief The eigenvectors of the real quasi-triangular \p t (row-major `n × n`, its 2 × 2 blocks standard) as the
     *        rows of `Xᵀ`, as LAPACK's `dtrevc`: real for a real eigenvalue, solved in real arithmetic; for a pair
     *        `re ± i im` (\p im `[k] > 0` the first) the first's vector, the second's its conjugate.
     */
    [[nodiscard]] inline real_eigenvectors real_vectors(const buffer<double>& t,
                                                        const buffer<double>& re,
                                                        const buffer<double>& im,
                                                        std::size_t           n)
    {
      using C = std::complex<double>;
      const double      smallest {std::numeric_limits<double>::min() * (static_cast<double>(n) / schur_ulp)};
      real_eigenvectors out      {buffer<double>(n * n, 0.0), buffer<double>(n * n, 0.0)};
      buffer<double>    x(n, 0.0);
      buffer<C>         y(n, C   {});
      for (std::size_t k {0}; k < n; ++k) {
        const double floor  {std::max(schur_ulp * (std::abs(re[k]) + std::abs(im[k])), smallest)};
        double*      row_re {out.re.data() + (k * n)};
        double*      row_im {out.im.data() + (k * n)};
        if (im[k] == 0.0) {
          x[k] = 1.0;
          solve_upward(t, n, k, k, re[k], floor, x);
          std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(k + 1), row_re);
        }
        else if (im[k] > 0.0) { // the pair's block is rows k, k + 1
          const double b {t[(k * n) + k + 1]};
          const double c {t[((k + 1) * n) + k]};
          y[k]     = std::abs(b) >= std::abs(c) ? C {1.0} : C {-im[k] / c};
          y[k + 1] = std::abs(b) >= std::abs(c) ? C {0.0, im[k] / b} : C {0.0, 1.0};
          solve_upward(t, n, k, k + 1, C {re[k], im[k]}, floor, y);
          for (std::size_t l {0}; l <= k + 1; ++l) {
            row_re[l]     = y[l].real();
            row_im[l]     = y[l].imag();
            row_re[l + n] = y[l].real();
            row_im[l + n] = -y[l].imag();
          }
        }
      }
      return out;
    }

    //! \brief `Vᵀ = Xᵀ Zᵀ`, row-major `n × n`, through the matrix product kernel.
    template <typename T>
    [[nodiscard]] buffer<T> back_transformed(const buffer<T>& xt,
                                             const buffer<T>& zt,
                                             std::size_t      n)
    {
      buffer<T> vt(n * n, T {});
      if (n > 0) {
        gemm(n, n, n, xt.data(), n, zt.data(), n, vt.data(), n);
      }
      return vt;
    }
  } // namespace detail

  /*!
   * \brief The eigenvalues, and optionally the right eigenvectors, of a general square matrix, as `numpy.linalg.eig`.
   *
   * `T` is `double` or `std::complex<double>`; the eigenvalues and vectors are complex either way (\ref real says
   * whether a real matrix's are all real). Column `j` of \ref vectors is the eigenvector of \ref values `[j]`,
   * `A V = V Λ`, each of unit norm with its component of largest modulus real.
   */
  template <typename T>
  class eigen_decomposition
  {
  public:

    /*!
     * \brief Decomposes \p input.
     * \param[in] input   A square matrix.
     * \param[in] vectors Whether to compute the eigenvectors (the eigenvalues alone cost a third as much).
     * \throws dimension_error   If \p input is not square.
     * \throws non_finite_error  If an entry is infinite or NaN.
     * \throws convergence_error If the QR iterations did not converge.
     */
    explicit eigen_decomposition(const matrix<T>& input,
                                 bool             vectors = true)
      : n_(input.rows())
    {
      if (input.cols() != n_) {
        throw dimension_error("eigendecomposition requires a square matrix");
      }
      detail::require_finite(input, "eigendecomposition");
      double largest {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < n_; ++j) {
          largest = std::max(largest, static_cast<double>(std::abs(input(i, j))));
        }
      }
      // A power of two near the largest magnitude: dividing by it is exact, and keeps the iterations' sums of squares
      // and the back-substitution's growth in range.
      const double   scale        {largest == 0.0 ? 1.0 : std::ldexp(1.0, std::ilogb(largest))};
      buffer<T>      a(n_ * n_, T {});
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < n_; ++j) {
          a[(i * n_) + j] = input(i, j) / detail::from_double<T>(scale);
        }
      }
      const detail::balancing       balanced {detail::balance(a, n_)};
      buffer<T>                     z {detail::hessenberg(a, n_, balanced.low, balanced.high, vectors)};
      const std::size_t             steps {detail::schur_steps_per_order * std::max<std::size_t>(10, balanced.high - balanced.low)};
      const detail::schur_target<T> target {&a, n_, balanced.low, balanced.high, vectors, vectors ? &z : nullptr}; // z is Zᵀ
      buffer<std::complex<double>>  values(n_);
      buffer<std::complex<double>>  vt;                                                                            // Vᵀ: each vector a row
      if constexpr (std::is_floating_point_v<T>) {
        buffer<double> re(n_, 0.0);
        buffer<double> im(n_, 0.0);
        for (std::size_t i {0}; i < n_; ++i) {
          re[i] = a[(i * n_) + i]; // the isolated ones; the active block's are overwritten
        }
        detail::real_schur(target, re, im, steps);
        for (std::size_t i {0}; i < n_; ++i) {
          values[i] = std::complex<double> {re[i] * scale, im[i] * scale};
          real_     = real_ && im[i] == 0.0;
        }
        if (vectors) {
          const detail::real_eigenvectors x   {detail::real_vectors(a, re, im, n_)};
          const buffer<double>            vre {detail::back_transformed(x.re, z, n_)};
          const buffer<double>            vim {real_ ? buffer<double>(n_ * n_, 0.0) : detail::back_transformed(x.im, z, n_)};
          vt = buffer<std::complex<double>>(n_ * n_);
          for (std::size_t e {0}; e < n_ * n_; ++e) {
            vt[e] = std::complex<double> {vre[e], vim[e]};
          }
        }
      }
      else {
        for (std::size_t i {0}; i < n_; ++i) {
          values[i] = a[(i * n_) + i];
        }
        detail::complex_schur(target, values, steps);
        for (std::size_t i {0}; i < n_; ++i) {
          values[i] *= scale;
        }
        real_ = false;
        if (vectors) {
          vt = detail::back_transformed(detail::complex_vectors(a, n_), z, n_);
        }
      }
      values_ = vector<std::complex<double>> {std::move(values)};
      if (!vectors) {
        return;
      }
      for (std::size_t c {0}; c < n_; ++c) { // Vᵀ ← Vᵀ D, then the exchanges undone, on its columns
        std::complex<double>* row {vt.data() + (c * n_)};
        for (std::size_t i {balanced.low}; i < balanced.high; ++i) {
          row[i] *= balanced.scale[i];
        }
        for (std::size_t s {balanced.swaps.size()}; s-- > 0;) {
          std::swap(row[balanced.swaps[s].first], row[balanced.swaps[s].second]);
        }
        finish(row, values_[c], T {});
      }
      buffer<std::complex<double>> v(n_ * n_);
      for (std::size_t r {0}; r < n_; ++r) {
        for (std::size_t c {0}; c < n_; ++c) {
          v[(r * n_) + c] = vt[(c * n_) + r];
        }
      }
      vectors_ = matrix<std::complex<double>> {n_, n_, std::move(v)};
    }

    //! \brief The eigenvalues, in the order of the Schur form.
    [[nodiscard]] const vector<std::complex<double>>& values() const noexcept
    {
      return values_;
    }

    //! \brief The eigenvectors as columns, ordered to match \ref values (empty where they were not computed).
    [[nodiscard]] const matrix<std::complex<double>>& vectors() const& noexcept
    {
      return vectors_;
    }

    //! \brief The eigenvectors, moved out of an expiring decomposition.
    [[nodiscard]] matrix<std::complex<double>> vectors() &&
    {
      return std::move(vectors_);
    }

    //! \brief Whether every eigenvalue is real (a real matrix's whose Schur form has no 2 × 2 block): its vectors
    //!        are then real too.
    [[nodiscard]] bool real() const noexcept
    {
      return real_;
    }

  private:

    //! \brief Normalises the eigenvector \p row (a row of `Vᵀ`) of a real matrix's eigenvalue \p value, as `dgeev`: a
    //!        real one to unit norm; a pair's first to unit norm, its largest component real, its second the first's
    //!        conjugate (the row above).
    void finish(std::complex<double>*       row,
                const std::complex<double>& value,
                double /*real*/) const
    {
      if (value.imag() == 0.0) {
        const double norm {detail::scaled_norm(row, n_)};
        for (std::size_t r {0}; r < n_; ++r) {
          row[r] = std::complex<double> {row[r].real() / norm, 0.0};
        }
      }
      else if (value.imag() < 0.0) {
        const std::complex<double>* first {row - n_};
        for (std::size_t r {0}; r < n_; ++r) {
          row[r] = std::conj(first[r]);
        }
      }
      else {
        detail::normalize_vector(row, n_, true);
      }
    }

    //! \brief Normalises the eigenvector \p row of a complex matrix, as `zgeev`: to unit norm, its largest component
    //!         real and positive.
    void finish(std::complex<double>* row,
                const std::complex<double>& /*value*/,
                const std::complex<double>& /*complex*/) const
    {
      detail::normalize_vector(row, n_, false);
    }

    std::size_t                    n_;           //!< The matrix order.
    vector<std::complex<double>>   values_;      //!< The eigenvalues.
    matrix<std::complex<double>>   vectors_;     //!< The eigenvectors as columns.
    bool                           real_ {true}; //!< Whether all eigenvalues are real.
  };

  /*!
   * \brief The eigenvalues alone of the square \p a, as `numpy.linalg.eigvals`.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the QR iterations did not converge.
   */
  template <typename T>
  [[nodiscard]] vector<std::complex<double>> eigenvalues(const matrix<T>& a)
  {
    return eigen_decomposition<T> {a, false}.values();
  }
} // namespace numerical

#endif // NUMERICAL_EIGEN_HPP
