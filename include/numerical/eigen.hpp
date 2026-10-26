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
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
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

    //! \brief `a · b`.
    template <typename A,
              typename B>
    [[nodiscard]] auto times(const A& a,
                             const B& b)
    {
      return a * b;
    }

    //! \brief `a · b` of two complex numbers by the schoolbook formula, without C Annex G's recovery of infinities, which
    //!        a large translation unit leaves out of line (a call per product: complex `eig` of order 64 took 2.7 times
    //!        as long in the Python binding as alone). The entries here are finite.
    [[nodiscard]] inline std::complex<double> times(const std::complex<double>& a,
                                                    const std::complex<double>& b)
    {
      return {(a.real() * b.real()) - (a.imag() * b.imag()), (a.real() * b.imag()) + (a.imag() * b.real())};
    }

    //! \brief `|x|` of a real `x`.
    [[nodiscard]] inline double largest_part(double x)
    {
      return std::abs(x);
    }

    //! \brief `max(|re x|, |im x|)`: within `√2` of `|x|`, without its square root.
    [[nodiscard]] inline double largest_part(const std::complex<double>& x)
    {
      return std::max(std::abs(x.real()), std::abs(x.imag()));
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
      buffer<T> gathered(out.high - out.low);
      for (bool unbalanced {true}; unbalanced;) {
        unbalanced = false;
        for (std::size_t i {out.low}; i < out.high; ++i) {
          for (std::size_t k {out.low}; k < out.high; ++k) { // the column gathered: scaled_norm's sums run contiguous
            gathered[k - out.low] = a[(k * n) + i];
          }
          double column       {scaled_norm(gathered)};
          double row          {scaled_norm(a.data() + (i * n) + out.low, out.high - out.low)};
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
            a[(i * n) + k] /= f;
          }
          for (std::size_t k {0}; k < out.high; ++k) {
            a[(k * n) + i] *= f;
          }
        }
      }
      return out;
    }

    /*!
     * \brief Reduces the active block `low … high − 1` of the balanced row-major `n × n` \p a to upper Hessenberg form
     *        by Householder reflections one column at a time, as LAPACK's `dgehd2`: `A = Q H Qᴴ`, the entries below the subdiagonal set
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
    [[nodiscard]] buffer<T> hessenberg_unblocked(buffer<T>&  a,
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
        v[0] += phase_of(v[0]) * from_carried<T>(norm);
        const T length    {from_carried<T>(scaled_norm(v))};
        for (T& element : v) {
          element /= length;
        }
        buffer<T> w(n - k, T {}); // the rows times (I − 2 v vᴴ) from the left: w = vᴴ A over columns k …
        for (std::size_t i {0}; i < v.size(); ++i) {
          const T  weight {conj_of(v[i])};
          const T* row    {a.data() + ((k + 1 + i) * n) + k};
          for (std::size_t c {0}; c < w.size(); ++c) {
            w[c] += times(weight, row[c]);
          }
        }
        for (std::size_t i {0}; i < v.size(); ++i) {
          const T twice {static_cast<T>(2) * v[i]};
          T*      row   {a.data() + ((k + 1 + i) * n) + k};
          for (std::size_t c {0}; c < w.size(); ++c) {
            row[c] -= times(twice, w[c]);
          }
        }
        for (std::size_t r {0}; r < high; ++r) { // A (I − 2 v vᴴ) on columns k + 1 …; rows below are zero there
          T* row {a.data() + (r * n) + k + 1};
          T  s   {};
          for (std::size_t j {0}; j < v.size(); ++j) {
            s += times(row[j], v[j]);
          }
          s *= static_cast<T>(2);
          for (std::size_t j {0}; j < v.size(); ++j) {
            row[j] -= times(s, conj_of(v[j]));
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
            w[c] += times(weight, row[c]);
          }
        }
        for (std::size_t i {0}; i < v.size(); ++i) {
          const T twice {static_cast<T>(2) * v[i]};
          T*      row   {q.data() + ((k + 1 + i) * n) + k + 1};
          for (std::size_t c {0}; c < w.size(); ++c) {
            row[c] -= times(twice, w[c]);
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

    //! \brief The columns of a panel of \ref hessenberg_blocked.
    inline constexpr std::size_t hessenberg_block {32};

    //! \brief The active order from which \ref hessenberg reduces by panels.
    inline constexpr std::size_t hessenberg_crossover {128};

    //! \brief Whether `(α, x)`, of tail norm \p rest, is already `(β, 0)` with `β` real: for a real one, its tail zero.
    [[nodiscard]] inline bool already_reduced(double /*alpha*/,
                                              double rest)
    {
      return rest == 0.0;
    }

    //! \brief The same for a complex one: its tail zero and `α` real.
    [[nodiscard]] inline bool already_reduced(const std::complex<double>& alpha,
                                              double                      rest)
    {
      return rest == 0.0 && alpha.imag() == 0.0;
    }

    /*!
     * \brief The Householder reflection `H = I − τ v vᴴ`, `v = (1, v₁…)`, with `Hᴴ (α, x) = (β, 0)` and `β` real, as
     *        LAPACK's `zlarfg` (`dlarfg` for a real `T`): \p alpha becomes `β` and the \p count entries at \p x become
     *        `v₁…`. \return `τ`, zero where `x` is zero and `α` real (the identity).
     */
    template <typename T>
    [[nodiscard]] T reflector_tau(T&          alpha,
                                  T*          x,
                                  std::size_t count)
    {
      const double rest {scaled_norm(x, count)};
      if (already_reduced(alpha, rest)) {
        return T {};
      }
      const double beta  {-std::copysign(std::hypot(static_cast<double>(std::abs(alpha)), rest), real_part(alpha))};
      const T      tau   {(from_carried<T>(beta) - alpha) / from_carried<T>(beta)};
      const T      scale {T {1} / (alpha - from_carried<T>(beta))};
      for (std::size_t i {0}; i < count; ++i) {
        x[i] *= scale;
      }
      alpha = from_carried<T>(beta);
      return tau;
    }

    //! \brief A real vector that rows are summed against by \ref dot_product.
    struct real_operand
    {
      buffer<double> v; //!< The vector.

      //! \brief `Σ row[l] v[l]`.
      [[nodiscard]] double dot(const double* row) const
      {
        return dot_product(row, v.data(), v.size());
      }
    };

    //! \brief A complex vector that rows are summed against, kept as `(re, −im)` and `(im, re)` pairs: a complex row, read
    //!        as its parts in order ([complex.numbers]), against each is the real or the imaginary part of the sum, one
    //!        \ref dot_product each.
    struct complex_operand
    {
      buffer<double> real_side;      //!< `(re, −im)` per entry.
      buffer<double> imaginary_side; //!< `(im, re)` per entry.

      //! \brief `Σ row[l] v[l]`.
      [[nodiscard]] std::complex<double> dot(const std::complex<double>* row) const
      {
        const auto* const parts {reinterpret_cast<const double*>(row)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        return {dot_product(parts, real_side.data(), real_side.size()), dot_product(parts, imaginary_side.data(), imaginary_side.size())};
      }
    };

    //! \brief The \p count entries \p stride apart at \p v as a \ref real_operand.
    [[nodiscard]] inline real_operand operand_of(const double* v,
                                                 std::size_t   stride,
                                                 std::size_t   count)
    {
      real_operand out {buffer<double>(count)};
      for (std::size_t l {0}; l < count; ++l) {
        out.v[l] = v[l * stride];
      }
      return out;
    }

    //! \brief The same as a \ref complex_operand.
    [[nodiscard]] inline complex_operand operand_of(const std::complex<double>* v,
                                                    std::size_t                 stride,
                                                    std::size_t                 count)
    {
      complex_operand out {buffer<double>(2 * count), buffer<double>(2 * count)};
      for (std::size_t l {0}; l < count; ++l) {
        const std::complex<double> x {v[l * stride]};
        out.real_side[2 * l]            = x.real();
        out.real_side[(2 * l) + 1]      = -x.imag();
        out.imaginary_side[2 * l]       = x.imag();
        out.imaginary_side[(2 * l) + 1] = x.real();
      }
      return out;
    }

    //! \brief One panel of \ref hessenberg_blocked, `Q_panel = I − V T Vᴴ`.
    template <typename T>
    struct hessenberg_panel
    {
      std::size_t first {0}; //!< The panel's first column; its reflectors act on rows first + 1 … high − 1.
      std::size_t width {0}; //!< Its columns.
      buffer<T>   v;         //!< `V`, row-major `(high − first − 1) × width`, unit lower trapezoidal.
      buffer<T>   t;         //!< `T`, row-major `width × width`, upper triangular.
    };

    /*!
     * \brief \ref hessenberg_unblocked by panels of \ref hessenberg_block columns, as LAPACK's `dgehrd` and `dlahr2`: a
     *        panel's reflectors are formed one column at a time, each column first updated by those before it through
     *        `Y = A V T`; the rest of the matrix then takes the panel from both sides through the matrix product
     *        kernel, `A ← (I − V Tᴴ Vᴴ)(A − Y Vᴴ)`. Its arguments and answer are \ref hessenberg_unblocked's.
     */
    template <typename T>
    [[nodiscard]] buffer<T> hessenberg_blocked(buffer<T>&  a,
                                               std::size_t n,
                                               std::size_t low,
                                               std::size_t high,
                                               bool        form)
    {
      std::vector<hessenberg_panel<T>> panels;
      for (std::size_t k {low}; k + 2 < high;) {
        const std::size_t b             {std::min(hessenberg_block, high - 2 - k)};
        const std::size_t m             {high - k - 1}; // the rows k + 1 … high − 1
        buffer<T>         v(m * b, T    {});
        buffer<T>         t(b * b, T    {});
        buffer<T>         y(high * b, T {});            // Y = A V T, row-major, rows 0 … high − 1
        buffer<T>         column(m, T   {});
        buffer<T>         w(b, T        {});
        for (std::size_t i {0}; i < b; ++i) {
          const std::size_t j {k + i};
          for (std::size_t r {0}; r < m; ++r) {
            column[r] = a[((k + 1 + r) * n) + j];
          }
          if (i > 0) {
            for (std::size_t r {0}; r < m; ++r) { // A Q: the column less Y Vᴴ e_j
              T sum {};
              for (std::size_t p {0}; p < i; ++p) {
                sum += times(y[((k + 1 + r) * b) + p], conj_of(v[((i - 1) * b) + p]));
              }
              column[r] -= sum;
            }
            for (std::size_t p {0}; p < i; ++p) { // then Qᴴ from the left: w = Tᴴ Vᴴ column, column −= V w
              T sum {};
              for (std::size_t r {p}; r < m; ++r) {
                sum += times(conj_of(v[(r * b) + p]), column[r]);
              }
              w[p] = sum;
            }
            for (std::size_t p {i}; p-- > 0;) {
              T sum {};
              for (std::size_t q {0}; q <= p; ++q) {
                sum += times(conj_of(t[(q * b) + p]), w[q]);
              }
              w[p] = sum;
            }
            for (std::size_t r {0}; r < m; ++r) {
              T sum {};
              for (std::size_t p {0}; p < std::min(i, r + 1); ++p) {
                sum += times(v[(r * b) + p], w[p]);
              }
              column[r] -= sum;
            }
          }
          T       alpha {column[i]};
          const T tau   {reflector_tau(alpha, column.data() + i + 1, m - i - 1)};
          v[(i * b) + i] = T {1};
          for (std::size_t r {i + 1}; r < m; ++r) {
            v[(r * b) + i] = column[r];
          }
          for (std::size_t r {0}; r < m; ++r) {
            a[((k + 1 + r) * n) + j] = r < i ? column[r] : (r == i ? alpha : T {});
          }
          // Y's column: τ (A v − Y Vᴴ v), A's columns past j as the panel found them, each row against v by dot_product
          const auto against_v {operand_of(v.data() + (i * b) + i, b, m - i)};
          parallel_for(m, high - j, [&](std::size_t r) {
                         y[((k + 1 + r) * b) + i] = against_v.dot(a.data() + ((k + 1 + r) * n) + j + 1);
                       });
          for (std::size_t p {0}; p < i; ++p) {
            T sum {};
            for (std::size_t r {i}; r < m; ++r) {
              sum += times(conj_of(v[(r * b) + p]), v[(r * b) + i]);
            }
            w[p] = sum;
          }
          for (std::size_t r {0}; r < m; ++r) {
            T sum {};
            for (std::size_t p {0}; p < i; ++p) {
              sum += times(y[((k + 1 + r) * b) + p], w[p]);
            }
            y[((k + 1 + r) * b) + i] = tau * (y[((k + 1 + r) * b) + i] - sum);
          }
          for (std::size_t p {0}; p < i; ++p) { // T's column: −τ T w
            T sum {};
            for (std::size_t q {p}; q < i; ++q) {
              sum += times(t[(p * b) + q], w[q]);
            }
            t[(p * b) + i] = -tau * sum;
          }
          t[(i * b) + i] = tau;
        }
        buffer<T> top((k + 1) * b, T {}); // Y's rows 0 … k: A(0 … k, k + 1 …) V T
        gemm(k + 1, b, m, a.data() + k + 1, n, v.data(), b, top.data(), b);
        for (std::size_t r {0}; r <= k; ++r) {
          for (std::size_t p {b}; p-- > 0;) {
            T sum {};
            for (std::size_t q {0}; q <= p; ++q) {
              sum += times(top[(r * b) + q], t[(q * b) + p]);
            }
            y[(r * b) + p] = sum;
          }
        }
        for (std::size_t r {0}; r <= k; ++r) { // the panel's columns above it: A − Y Vᴴ
          for (std::size_t i {1}; i < b; ++i) {
            T sum {};
            for (std::size_t p {0}; p < i; ++p) {
              sum += times(y[(r * b) + p], conj_of(v[((i - 1) * b) + p]));
            }
            a[(r * n) + k + i] -= sum;
          }
        }
        const std::size_t rest {high - k - b}; // the columns k + b … high − 1, from the right
        buffer<T>         minus_y(high * b);
        for (std::size_t e {0}; e < high * b; ++e) {
          minus_y[e] = -y[e];
        }
        buffer<T> v_adjoint(b * m);
        for (std::size_t r {0}; r < m; ++r) {
          for (std::size_t p {0}; p < b; ++p) {
            v_adjoint[(p * m) + r] = conj_of(v[(r * b) + p]);
          }
        }
        gemm(high, rest, b, minus_y.data(), b, v_adjoint.data() + b - 1, m, a.data() + k + b, n);
        const std::size_t columns                {n - k - b}; // then rows k + 1 … high − 1, columns k + b …, from the left
        buffer<T>         product(b * columns, T {});
        gemm(b, columns, m, v_adjoint.data(), m, a.data() + ((k + 1) * n) + k + b, n, product.data(), columns);
        buffer<T> scaled(b * columns, T          {});
        for (std::size_t p {0}; p < b; ++p) {
          for (std::size_t q {0}; q <= p; ++q) {
            const T factor {-conj_of(t[(q * b) + p])};
            for (std::size_t c {0}; c < columns; ++c) {
              scaled[(p * columns) + c] += times(factor, product[(q * columns) + c]);
            }
          }
        }
        gemm(m, columns, b, v.data(), b, scaled.data(), columns, a.data() + ((k + 1) * n) + k + b, n);
        if (form) {
          panels.push_back({k, b, std::move(v), std::move(t)});
        }
        k += b;
      }
      if (!form) {
        return {};
      }
      buffer<T> q(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        q[(i * n) + i] = T {1};
      }
      for (std::size_t s {panels.size()}; s-- > 0;) { // Q = Q₁ … Q_p, backward: Q ← (I − V T Vᴴ) Q on rows first + 1 …
        const hessenberg_panel<T>& panel {panels[s]};
        const std::size_t          m     {high - panel.first - 1};
        const std::size_t          b     {panel.width};
        T* const                   block {q.data() + ((panel.first + 1) * n) + panel.first + 1};
        buffer<T>                  v_adjoint(b * m);
        for (std::size_t r {0}; r < m; ++r) {
          for (std::size_t p {0}; p < b; ++p) {
            v_adjoint[(p * m) + r] = conj_of(panel.v[(r * b) + p]);
          }
        }
        buffer<T> product(b * m, T {});
        gemm(b, m, m, v_adjoint.data(), m, block, n, product.data(), m);
        buffer<T> scaled(b * m, T  {});
        for (std::size_t p {0}; p < b; ++p) {
          for (std::size_t r {p}; r < b; ++r) {
            const T factor {-panel.t[(p * b) + r]};
            for (std::size_t c {0}; c < m; ++c) {
              scaled[(p * m) + c] += times(factor, product[(r * m) + c]);
            }
          }
        }
        gemm(m, m, b, panel.v.data(), b, scaled.data(), m, block, n);
      }
      buffer<T> qt(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          qt[(j * n) + i] = q[(i * n) + j];
        }
      }
      return qt;
    }

    //! \brief \ref hessenberg_blocked from an active order of \ref hessenberg_crossover, else \ref hessenberg_unblocked.
    template <typename T>
    [[nodiscard]] buffer<T> hessenberg(buffer<T>&  a,
                                       std::size_t n,
                                       std::size_t low,
                                       std::size_t high,
                                       bool        form)
    {
      if (high - low >= hessenberg_crossover) {
        return hessenberg_blocked(a, n, low, high, form);
      }
      return hessenberg_unblocked(a, n, low, high, form);
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

    /*!
     * \brief A reflection `I − τ v vᴴ`, `v = (1, v₂, v₃)`, on the entries \p from … \p to − 1 of three rows: its adjoint
     *        from the left with \p adjoint (rows of `H`), else from the right (rows of `Zᵀ`, which are columns of `Z`).
     *        A reflection of two passes \p r3 = \p r2 and \p v3 = 0. One loop over the entries, which vectorises.
     */
    template <typename T>
    void reflect_three_rows(T*          r1,
                            T*          r2,
                            T*          r3,
                            std::size_t from,
                            std::size_t to,
                            const T&    tau,
                            const T&    v2,
                            const T&    v3,
                            bool        adjoint)
    {
      const T f  {adjoint ? conj_of(tau) : tau};
      const T a2 {adjoint ? conj_of(v2) : v2};
      const T a3 {adjoint ? conj_of(v3) : v3};
      const T b2 {adjoint ? v2 : conj_of(v2)};
      const T b3 {adjoint ? v3 : conj_of(v3)};
      for (std::size_t j {from}; j < to; ++j) {
        const T sum {times(f, r1[j] + times(a2, r2[j]) + times(a3, r3[j]))};
        r1[j] -= sum;
        r2[j] -= times(sum, b2);
        r3[j] -= times(sum, b3);
      }
    }

    //! \brief \ref reflect_three_rows for a reflection of two entries, `v = (1, v₂)`, on two rows: its own loop, which
    //!        vectorises where the third row aliasing the second would not.
    template <typename T>
    void reflect_two_rows(T*          r1,
                          T*          r2,
                          std::size_t from,
                          std::size_t to,
                          const T&    tau,
                          const T&    v2,
                          bool        adjoint)
    {
      const T f  {adjoint ? conj_of(tau) : tau};
      const T a2 {adjoint ? conj_of(v2) : v2};
      const T b2 {adjoint ? v2 : conj_of(v2)};
      for (std::size_t j {from}; j < to; ++j) {
        const T sum {times(f, r1[j] + times(a2, r2[j]))};
        r1[j] -= sum;
        r2[j] -= times(sum, b2);
      }
    }

    /*!
     * \brief A reflection `I − τ v vᴴ`, `v = (1, v₂, v₃)`, from the right on \p rows rows (\p ld apart) of the three
     *        columns at \p first: `x ← x − τ (x₁ + v₂ x₂ + v₃ x₃) (1, v̄₂, v̄₃)` on each. The rows are short and apart:
     *        one call for them all.
     */
    template <typename T>
    void reflect_three_columns(T*          first,
                               std::size_t rows,
                               std::size_t ld,
                               const T&    tau,
                               const T&    v2,
                               const T&    v3)
    {
      const T w2 {conj_of(v2)};
      const T w3 {conj_of(v3)};
      for (std::size_t r {0}; r < rows; ++r) {
        T* const x   {first + (r * ld)};
        const T  sum {times(tau, x[0] + times(v2, x[1]) + times(v3, x[2]))};
        x[0] -= sum;
        x[1] -= times(sum, w2);
        x[2] -= times(sum, w3);
      }
    }

    //! \brief The same for a reflection of two entries, `v = (1, v₂)`, on two columns.
    template <typename T>
    void reflect_two_columns(T*          first,
                             std::size_t rows,
                             std::size_t ld,
                             const T&    tau,
                             const T&    v2)
    {
      const T w2 {conj_of(v2)};
      for (std::size_t r {0}; r < rows; ++r) {
        T* const x   {first + (r * ld)};
        const T  sum {times(tau, x[0] + times(v2, x[1]))};
        x[0] -= sum;
        x[1] -= times(sum, w2);
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
            reflect_two_rows(&at(k2, 0), &at(k2 + 1, 0), k2, last, t1, v2, true);
            reflect_two_columns(&at(first, k2), std::min(k2 + 2, i) - first + 1, n, t1, v2);
            if (t.zt != nullptr) {
              C* const r1 {t.zt->data() + (k2 * n)};
              reflect_two_rows(r1, r1 + n, t.low, t.high, t1, v2, false);
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

    //! \brief A signed index from 1, as LAPACK's: the multishift sweep starts its bulges in phantom columns above the
    //!        matrix, before its first row.
    using lapack_index = std::ptrdiff_t;

    //! \brief A row-major matrix read through indices from 1, `(1, 1)` its first entry.
    template <typename T>
    struct one_based
    {
      //! \brief The matrix whose first entry is at \p first, its rows \p distance apart.
      one_based(T*           first,
                lapack_index distance)
        : data(first), ld(distance)
      {}

      T*           data; //!< The first entry.
      lapack_index ld;   //!< The distance between rows.

      //! \brief The entry of row \p i and column \p j (from 1).
      [[nodiscard]] T& operator()(lapack_index i,
                                  lapack_index j) const
      {
        return data[((i - 1) * ld) + (j - 1)];
      }
    };

    //! \brief A count as an unsigned size (never negative where it is taken).
    [[nodiscard]] inline std::size_t count_of(lapack_index count)
    {
      return static_cast<std::size_t>(count);
    }

    //! \brief LAPACK's `nmin`: the active order below which `dlaqr0`'s iterations skip a sweep after enough
    //!        deflation.
    inline constexpr lapack_index multishift_crossover {75};

    //! \brief The order from which a real Schur form is taken by \ref multishift_schur, below which \ref real_schur is
    //!        faster (eight arm64 cores: order 128 1.85 against 2.60 ms; 224, 8.9 against 8.6; with the vectors, 192,
    //!        8.1 against 7.1).
    [[nodiscard]] constexpr lapack_index multishift_from(double /*real*/)
    {
      return 200;
    }

    //! \brief The same for a complex one (order 76 1.8 against 2.1 ms, 128 7.5 against 6.5; with the vectors, 100, 6.4
    //!        against 5.6).
    [[nodiscard]] constexpr lapack_index multishift_from(const std::complex<double>& /*complex*/)
    {
      return 100;
    }

    //! \brief The iterations without deflation after which the window grows (`kexnw`).
    inline constexpr lapack_index window_growth {5};

    //! \brief The iterations without deflation after which the shifts are exceptional (`kexsh`).
    inline constexpr lapack_index exceptional_after {6};

    //! \brief The share of the window, in percent, whose deflation makes the sweep unnecessary (`nibble`).
    inline constexpr lapack_index nibble {14};

    //! \brief A plane rotation `[c s; −s̄ c]` and what it leaves of its first argument.
    template <typename T>
    struct plane_rotation
    {
      double c {1.0}; //!< The cosine.
      T      s {};    //!< The sine.
      T      r {};    //!< `c f + s g`.
    };

    //! \brief The rotation that takes `(f, g)` to `(r, 0)`, as LAPACK's `dlartg`: `r` of `f`'s sign.
    [[nodiscard]] inline plane_rotation<double> givens(double f,
                                                       double g)
    {
      if (g == 0.0) {
        return {1.0, 0.0, f};
      }
      if (f == 0.0) {
        return {0.0, std::copysign(1.0, g), std::abs(g)};
      }
      const double r {std::copysign(std::hypot(f, g), f)};
      return {std::abs(f) / std::abs(r), g / r, r};
    }

    //! \brief The rotation that takes `(f, g)` to `(r, 0)`, as LAPACK's `zlartg`: `c` real, `r` of `f`'s phase.
    [[nodiscard]] inline plane_rotation<std::complex<double>> givens(const std::complex<double>& f,
                                                                     const std::complex<double>& g)
    {
      using C = std::complex<double>;
      if (g == C {}) {
        return {1.0, C {}, f};
      }
      if (f == C {}) {
        return {0.0, std::conj(g) / std::abs(g), C {std::abs(g)}};
      }
      const double size  {std::abs(f)};
      const double d     {std::hypot(size, std::abs(g))};
      const C      phase {f / size};
      return {size / d, phase * std::conj(g) / d, phase * d};
    }

    //! \brief `(x, y) ← (c x + s y, c y − s̄ x)` over \p count pairs \p stride apart.
    template <typename T>
    void rotate_pair(T*          x,
                     T*          y,
                     std::size_t count,
                     std::size_t stride,
                     double      c,
                     const T&    s)
    {
      for (std::size_t i {0}; i < count; ++i) {
        const T xi {x[i * stride]};
        const T yi {y[i * stride]};
        x[i * stride] = (c * xi) + times(s, yi);
        y[i * stride] = (c * yi) - times(conj_of(s), xi);
      }
    }

    /*!
     * \brief Exchanges the diagonal entries \p k and \p k + 1 of the upper triangular (or quasi-triangular, both 1 × 1
     *        there) \p t by a plane rotation, as LAPACK's `ztrexc` step and `dlaexc` for two 1 × 1 blocks; the rows of
     *        \p qt (`Qᵀ`) turn with it.
     */
    template <typename T>
    void exchange_diagonal(one_based<T>  t,
                           one_based<T>  qt,
                           lapack_index  n,
                           lapack_index  k)
    {
      const T                 t11 {t(k, k)};
      const T                 t22 {t(k + 1, k + 1)};
      const plane_rotation<T> g   {givens(t(k, k + 1), t22 - t11)};
      if (k + 2 <= n) {
        rotate_pair(&t(k, k + 2), &t(k + 1, k + 2), count_of(n - k - 1), 1, g.c, g.s);
      }
      rotate_pair(&t(1, k), &t(1, k + 1), count_of(k - 1), count_of(t.ld), g.c, conj_of(g.s));
      t(k, k)         = t22;
      t(k + 1, k + 1) = t11;
      rotate_pair(&qt(k, 1), &qt(k + 1, 1), count_of(n), 1, g.c, conj_of(g.s));
    }

    //! \brief `H = I − τ u uᵀ` of three entries applied from the left to the 3 × \p count rows at \p t (row stride \p ld).
    inline void reflect_rows_3(const std::array<double, 3>& u,
                               double                       tau,
                               double                *      t,
                               std::size_t                  ld,
                               std::size_t                  count)
    {
      for (std::size_t c {0}; c < count; ++c) {
        const double sum {(u[0] * t[c]) + (u[1] * t[ld + c]) + (u[2] * t[(2 * ld) + c])};
        t[c]            -= tau * sum * u[0];
        t[ld + c]       -= tau * sum * u[1];
        t[(2 * ld) + c] -= tau * sum * u[2];
      }
    }

    //! \brief The same from the right, to the \p count × 3 columns at \p t.
    inline void reflect_columns_3(const std::array<double, 3>& u,
                                  double                       tau,
                                  double                *      t,
                                  std::size_t                  ld,
                                  std::size_t                  count)
    {
      for (std::size_t r {0}; r < count; ++r) {
        double* const row {t + (r * ld)};
        const double  sum {(row[0] * u[0]) + (row[1] * u[1]) + (row[2] * u[2])};
        row[0] -= tau * sum * u[0];
        row[1] -= tau * sum * u[1];
        row[2] -= tau * sum * u[2];
      }
    }

    /*!
     * \brief The solution `X` (\p n1 × \p n2, row-major in \p x) of `TL X − X TR = B`, \p n1 and \p n2 at most 2, by
     *        Gaussian elimination with complete pivoting on its Kronecker form, a pivot below `ε` times the largest
     *        coefficient taken as that, as LAPACK's `dlasy2` (the blocks' entries near 1: no scaling against overflow).
     */
    inline void small_sylvester(const one_based<double>& tl,
                                const one_based<double>& tr,
                                const one_based<double>& b,
                                lapack_index             n1,
                                lapack_index             n2,
                                double (&x)[4])
    {
      const std::size_t size    {count_of(n1 * n2)};
      double            k[4][4] {};
      double            rhs[4]  {};
      double            largest {0.0};
      for (lapack_index i {1}; i <= n1; ++i) {
        for (lapack_index j {1}; j <= n2; ++j) {
          const std::size_t row {count_of(((i - 1) * n2) + (j - 1))};
          rhs[row] = b(i, j);
          for (lapack_index p {1}; p <= n1; ++p) {
            k[row][count_of(((p - 1) * n2) + (j - 1))] += tl(i, p);
          }
          for (lapack_index q {1}; q <= n2; ++q) {
            k[row][count_of(((i - 1) * n2) + (q - 1))] -= tr(q, j);
          }
        }
      }
      for (std::size_t r {0}; r < size; ++r) {
        for (std::size_t c {0}; c < size; ++c) {
          largest = std::max(largest, std::abs(k[r][c]));
        }
      }
      const double floor        {std::max(schur_ulp * largest, std::numeric_limits<double>::min())};
      std::size_t  column_of[4] {0, 1, 2, 3};
      for (std::size_t s {0}; s < size; ++s) {
        std::size_t pr {s};
        std::size_t pc {s};
        for (std::size_t r {s}; r < size; ++r) {
          for (std::size_t c {s}; c < size; ++c) {
            if (std::abs(k[r][c]) > std::abs(k[pr][pc])) {
              pr = r;
              pc = c;
            }
          }
        }
        std::swap(k[s], k[pr]);
        std::swap(rhs[s], rhs[pr]);
        for (std::size_t r {0}; r < size; ++r) {
          std::swap(k[r][s], k[r][pc]);
        }
        std::swap(column_of[s], column_of[pc]);
        if (std::abs(k[s][s]) < floor) {
          k[s][s] = floor;
        }
        for (std::size_t r {s + 1}; r < size; ++r) {
          const double factor {k[r][s] / k[s][s]};
          for (std::size_t c {s}; c < size; ++c) {
            k[r][c] -= factor * k[s][c];
          }
          rhs[r] -= factor * rhs[s];
        }
      }
      double y[4] {};
      for (std::size_t s {size}; s-- > 0;) {
        double sum {rhs[s]};
        for (std::size_t c {s + 1}; c < size; ++c) {
          sum -= k[s][c] * y[c];
        }
        y[s] = sum / k[s][s];
      }
      for (std::size_t s {0}; s < size; ++s) {
        x[column_of[s]] = y[s];
      }
    }

    //! \brief The standard form of the 2 × 2 block at \p j of the quasi-triangular \p t, its rotation carried to the rest
    //!        of `T` and to the rows of \p qt, as `dlaexc` ends.
    inline void standardize_at(one_based<double> t,
                               one_based<double> qt,
                               lapack_index      n,
                               lapack_index      j)
    {
      const standard_block block {standardize_block(t(j, j), t(j, j + 1), t(j + 1, j), t(j + 1, j + 1))};
      if (j + 2 <= n) {
        rotate(&t(j, j + 2), &t(j + 1, j + 2), count_of(n - j - 1), 1, block.cs, block.sn);
      }
      rotate(&t(1, j), &t(1, j + 1), count_of(j - 1), count_of(t.ld), block.cs, block.sn);
      rotate(&qt(j, 1), &qt(j + 1, 1), count_of(n), 1, block.cs, block.sn);
    }

    /*!
     * \brief Exchanges the adjacent diagonal blocks of the real quasi-triangular \p t at \p j1, of orders \p n1 then
     *        \p n2, by an orthogonal similarity carried to the rows of \p qt, as LAPACK's `dlaexc`. \return Whether it
     *        did: a swap that would perturb the blocks past `10 ε ‖D‖` is refused and nothing changes.
     */
    inline bool exchange_blocks(one_based<double> t,
                                one_based<double> qt,
                                lapack_index      n,
                                lapack_index      j1,
                                lapack_index      n1,
                                lapack_index      n2)
    {
      if (n1 == 1 && n2 == 1) {
        exchange_diagonal(t, qt, n, j1);
        return true;
      }
      const lapack_index      j2    {j1 + 1};
      const lapack_index      j3    {j1 + 2};
      const lapack_index      j4    {j1 + 3};
      const lapack_index      nd    {n1 + n2};
      std::array<double, 16>  d     {};
      const one_based<double> dd    {d.data(), 4};
      double                  norm  {0.0};
      for (lapack_index i {1}; i <= nd; ++i) {
        for (lapack_index j {1}; j <= nd; ++j) {
          dd(i, j) = t(j1 + i - 1, j1 + j - 1);
          norm     = std::max(norm, std::abs(dd(i, j)));
        }
      }
      const double thresh {std::max(10.0 * schur_ulp * norm, std::numeric_limits<double>::min() / schur_ulp)};
      double       x[4] {};
      small_sylvester(dd, one_based<double> {&dd(n1 + 1, n1 + 1), 4}, one_based<double> {&dd(1, n1 + 1), 4}, n1, n2, x);
      const auto            at {[&x, n2](lapack_index i, lapack_index j) { return x[count_of(((i - 1) * n2) + (j - 1))]; }};
      const std::size_t     tld     {count_of(t.ld)};
      std::array<double, 3> u1      {};
      std::array<double, 3> u2      {};
      double                tau1    {};
      double                tau2    {};
      double                residue {};
      const double          t11     {t(j1, j1)};
      const double          t33     {t(j1 + nd - 1, j1 + nd - 1)};
      if (n1 == 1) { // and n2 == 2: one reflection, tried on the block first
        u1[0] = 1.0;
        u1[1] = at(1, 1);
        u1[2] = at(1, 2);
        double alpha {u1[2]};
        tau1  = reflector_tau(alpha, u1.data(), 2);
        u1[2] = 1.0;
        reflect_rows_3(u1, tau1, d.data(), 4, 3);
        reflect_columns_3(u1, tau1, d.data(), 4, 3);
        residue = std::max({std::abs(dd(3, 1)), std::abs(dd(3, 2)), std::abs(dd(3, 3) - t11)});
      }
      else if (n2 == 1) { // n1 == 2
        u1[0] = -at(1, 1);
        u1[1] = -at(2, 1);
        u1[2] = 1.0;
        double alpha {u1[0]};
        tau1  = reflector_tau(alpha, u1.data() + 1, 2);
        u1[0] = 1.0;
        reflect_rows_3(u1, tau1, d.data(), 4, 3);
        reflect_columns_3(u1, tau1, d.data(), 4, 3);
        residue = std::max({std::abs(dd(2, 1)), std::abs(dd(3, 1)), std::abs(dd(1, 1) - t33)});
      }
      else { // two 2 × 2 blocks: two reflections
        u1[0] = -at(1, 1);
        u1[1] = -at(2, 1);
        u1[2] = 1.0;
        double alpha1 {u1[0]};
        tau1  = reflector_tau(alpha1, u1.data() + 1, 2);
        u1[0] = 1.0;
        const double temp {-tau1 * (at(1, 2) + (u1[1] * at(2, 2)))};
        u2[0] = (-temp * u1[1]) - at(2, 2);
        u2[1] = -temp * u1[2];
        u2[2] = 1.0;
        double alpha2 {u2[0]};
        tau2  = reflector_tau(alpha2, u2.data() + 1, 2);
        u2[0] = 1.0;
        reflect_rows_3(u1, tau1, d.data(), 4, 4);
        reflect_columns_3(u1, tau1, d.data(), 4, 4);
        reflect_rows_3(u2, tau2, &dd(2, 1), 4, 4);
        reflect_columns_3(u2, tau2, &dd(1, 2), 4, 4);
        residue = std::max({std::abs(dd(3, 1)), std::abs(dd(3, 2)), std::abs(dd(4, 1)), std::abs(dd(4, 2))});
      }
      if (residue > thresh) { // the swap would not leave the blocks apart: refused, T as it was
        return false;
      }
      if (n1 == 1) {
        reflect_rows_3(u1, tau1, &t(j1, j1), tld, count_of(n - j1 + 1));
        reflect_columns_3(u1, tau1, &t(1, j1), tld, count_of(j2));
        t(j3, j1) = 0.0;
        t(j3, j2) = 0.0;
        t(j3, j3) = t11;
        reflect_rows_3(u1, tau1, &qt(j1, 1), count_of(qt.ld), count_of(n));
      }
      else if (n2 == 1) {
        reflect_columns_3(u1, tau1, &t(1, j1), tld, count_of(j3));
        reflect_rows_3(u1, tau1, &t(j1, j2), tld, count_of(n - j1));
        t(j1, j1) = t33;
        t(j2, j1) = 0.0;
        t(j3, j1) = 0.0;
        reflect_rows_3(u1, tau1, &qt(j1, 1), count_of(qt.ld), count_of(n));
      }
      else {
        reflect_rows_3(u1, tau1, &t(j1, j1), tld, count_of(n - j1 + 1));
        reflect_columns_3(u1, tau1, &t(1, j1), tld, count_of(j4));
        reflect_rows_3(u2, tau2, &t(j2, j1), tld, count_of(n - j1 + 1));
        reflect_columns_3(u2, tau2, &t(1, j2), tld, count_of(j4));
        t(j3, j1) = 0.0;
        t(j3, j2) = 0.0;
        t(j4, j1) = 0.0;
        t(j4, j2) = 0.0;
        reflect_rows_3(u1, tau1, &qt(j1, 1), count_of(qt.ld), count_of(n));
        reflect_rows_3(u2, tau2, &qt(j2, 1), count_of(qt.ld), count_of(n));
      }
      if (n2 == 2) {
        standardize_at(t, qt, n, j1);
      }
      if (n1 == 2) {
        standardize_at(t, qt, n, j1 + n2);
      }
      return true;
    }

    //! \brief The order of the diagonal block of the quasi-triangular \p t that starts at \p j: 2 where a subdiagonal
    //!        entry below it is nonzero.
    [[nodiscard]] inline lapack_index block_order(one_based<double> t,
                                                  lapack_index      n,
                                                  lapack_index      j)
    {
      return j < n && t(j + 1, j) != 0.0 ? 2 : 1;
    }

    /*!
     * \brief Moves the diagonal block of the real quasi-triangular \p t at \p ifst to \p ilst by exchanges of adjacent
     *        blocks, the rows of \p qt (`Qᵀ`) turning with it, as LAPACK's `dtrexc`. \p ilst becomes where the block
     *        ended: short of its goal where an exchange was refused (then \return false).
     */
    inline bool reorder(one_based<double> t,
                        one_based<double> qt,
                        lapack_index      n,
                        lapack_index      ifst,
                        lapack_index&     ilst)
    {
      if (ifst > 1 && t(ifst, ifst - 1) != 0.0) {
        --ifst;
      }
      lapack_index nbf {block_order(t, n, ifst)};
      if (ilst > 1 && t(ilst, ilst - 1) != 0.0) {
        --ilst;
      }
      const lapack_index nbl {block_order(t, n, ilst)};
      if (ifst == ilst) {
        return true;
      }
      struct refused : std::exception // an exchange refused: the block stays where it got to
      {};
      const auto   exchange {[&t, &qt, n](lapack_index j1, lapack_index n1, lapack_index n2) {
                               if (!exchange_blocks(t, qt, n, j1, n1, n2)) {
                                 throw refused {};
                               }
                             }};
      lapack_index here {ifst};
      try {
        if (ifst < ilst) {
          if (nbf == 2 && nbl == 1) {
            --ilst;
          }
          if (nbf == 1 && nbl == 2) {
            ++ilst;
          }
          while (here < ilst) {
            if (nbf != 3) { // the block, 1 × 1 or 2 × 2, past the next one
              const lapack_index next {here + nbf + 1 <= n && t(here + nbf + 1, here + nbf) != 0.0 ? 2 : 1};
              exchange(here, nbf, next);
              here += next;
              if (nbf == 2 && t(here + 1, here) == 0.0) {
                nbf = 3; // the 2 × 2 block split into two 1 × 1
              }
            }
            else { // two 1 × 1 blocks, each past the next block
              const lapack_index next {here + 3 <= n && t(here + 3, here + 2) != 0.0 ? 2 : 1};
              exchange(here + 1, 1, next);
              if (next == 2 && t(here + 2, here + 1) == 0.0) { // the next block split too: the first past both pieces
                exchange(here, 1, 1);
                exchange(here + 1, 1, 1);
                here += 2;
              }
              else {
                exchange(here, 1, next);
                here += next;
              }
            }
          }
        }
        else {
          while (here > ilst) {
            if (nbf != 3) {
              const lapack_index next {here >= 3 && t(here - 1, here - 2) != 0.0 ? 2 : 1};
              exchange(here - next, next, nbf);
              here -= next;
              if (nbf == 2 && t(here + 1, here) == 0.0) {
                nbf = 3;
              }
            }
            else {
              const lapack_index next {here >= 3 && t(here - 1, here - 2) != 0.0 ? 2 : 1};
              exchange(here - next, next, 1);
              if (next == 2 && t(here, here - 1) == 0.0) {
                exchange(here, 1, 1);
                exchange(here - 1, 1, 1);
                here -= 2;
              }
              else {
                exchange(here - next + 1, next, 1);
                here -= next;
              }
            }
          }
        }
      }
      catch (const refused&) {
        ilst = here;
        return false;
      }
      ilst = here;
      return true;
    }

    //! \brief Moves the diagonal entry of the complex upper triangular \p t at \p ifst to \p ilst, as LAPACK's `ztrexc`.
    //!        \return true: exchanges of 1 × 1 blocks never fail.
    inline bool reorder(one_based<std::complex<double>> t,
                        one_based<std::complex<double>> qt,
                        lapack_index                    n,
                        lapack_index                    ifst,
                        lapack_index&                   ilst)
    {
      for (lapack_index k {ifst}; k < ilst; ++k) {
        exchange_diagonal(t, qt, n, k);
      }
      for (lapack_index k {ifst - 1}; k >= ilst; --k) {
        exchange_diagonal(t, qt, n, k);
      }
      return true;
    }

    /*!
     * \brief A multiple of the first column of `(H − s₁ I)(H − s₂ I)` for the leading 2 × 2 or 3 × 3 (\p count) of the
     *        real Hessenberg \p h, the shifts a conjugate pair or two reals, as LAPACK's `dlaqr1`.
     */
    inline void bulge_column(one_based<double>           h,
                             lapack_index                count,
                             const std::complex<double>& s1,
                             const std::complex<double>& s2,
                             double*                     v)
    {
      const double sr1 {s1.real()};
      const double si1 {s1.imag()};
      const double sr2 {s2.real()};
      const double si2 {s2.imag()};
      if (count == 2) {
        const double s {std::abs(h(1, 1) - sr2) + std::abs(si2) + std::abs(h(2, 1))};
        if (s == 0.0) {
          v[0] = 0.0;
          v[1] = 0.0;
          return;
        }
        const double h21s {h(2, 1) / s};
        v[0] = (h21s * h(1, 2)) + ((h(1, 1) - sr1) * ((h(1, 1) - sr2) / s)) - (si1 * (si2 / s));
        v[1] = h21s * (h(1, 1) + h(2, 2) - sr1 - sr2);
        return;
      }
      const double s {std::abs(h(1, 1) - sr2) + std::abs(si2) + std::abs(h(2, 1)) + std::abs(h(3, 1))};
      if (s == 0.0) {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
        return;
      }
      const double h21s {h(2, 1) / s};
      const double h31s {h(3, 1) / s};
      v[0] = ((h(1, 1) - sr1) * ((h(1, 1) - sr2) / s)) - (si1 * (si2 / s)) + (h(1, 2) * h21s) + (h(1, 3) * h31s);
      v[1] = (h21s * (h(1, 1) + h(2, 2) - sr1 - sr2)) + (h(2, 3) * h31s);
      v[2] = (h31s * (h(1, 1) + h(3, 3) - sr1 - sr2)) + (h21s * h(3, 2));
    }

    //! \brief The same for a complex \p h and any two shifts, as LAPACK's `zlaqr1`.
    inline void bulge_column(one_based<std::complex<double>> h,
                             lapack_index                    count,
                             const std::complex<double>&     s1,
                             const std::complex<double>&     s2,
                             std::complex<double>*           v)
    {
      using C = std::complex<double>;
      if (count == 2) {
        const double s {modulus_1(h(1, 1) - s2) + modulus_1(h(2, 1))};
        if (s == 0.0) {
          v[0] = C {};
          v[1] = C {};
          return;
        }
        const C h21s {h(2, 1) / s};
        v[0] = (h21s * h(1, 2)) + ((h(1, 1) - s1) * ((h(1, 1) - s2) / s));
        v[1] = h21s * (h(1, 1) + h(2, 2) - s1 - s2);
        return;
      }
      const double s {modulus_1(h(1, 1) - s2) + modulus_1(h(2, 1)) + modulus_1(h(3, 1))};
      if (s == 0.0) {
        v[0] = C {};
        v[1] = C {};
        v[2] = C {};
        return;
      }
      const C h21s {h(2, 1) / s};
      const C h31s {h(3, 1) / s};
      v[0] = ((h(1, 1) - s1) * ((h(1, 1) - s2) / s)) + (h21s * h(1, 2)) + (h31s * h(1, 3));
      v[1] = (h21s * (h(1, 1) + h(2, 2) - s1 - s2)) + (h31s * h(2, 3));
      v[2] = (h31s * (h(1, 1) + h(3, 3) - s1 - s2)) + (h21s * h(3, 2));
    }

    //! \brief Arranges a real matrix's shifts so that each pair is a conjugate pair or two reals, as `dlaqr5` begins
    //!        (conjugate shifts already adjacent).
    inline void pair_shifts(std::complex<double>* s,
                            lapack_index          count,
                            double /*real*/)
    {
      for (lapack_index i {0}; i + 2 < count; i += 2) {
        if (s[i].imag() != -s[i + 1].imag()) {
          const std::complex<double> first {s[i]};
          s[i]     = s[i + 1];
          s[i + 1] = s[i + 2];
          s[i + 2] = first;
        }
      }
    }

    //! \brief A complex matrix's shifts pair as they come.
    inline void pair_shifts(std::complex<double>* /*s*/,
                            lapack_index /*count*/,
                            const std::complex<double>& /*complex*/)
    {
    }

    //! \brief The shift \p s as the matrix's scalar: a real matrix's are real where they are used one by one.
    [[nodiscard]] inline double as_scalar(const std::complex<double>& s,
                                          double /*real*/)
    {
      return s.real();
    }

    //! \brief A complex matrix's shift as it is.
    [[nodiscard]] inline std::complex<double> as_scalar(const std::complex<double>& s,
                                                        const std::complex<double>& /*complex*/)
    {
      return s;
    }

    //! \brief `H(k + 1, k)` set to zero where the deflation criterion of Ahues and Tisseur holds, as the sweep of
    //!        `dlaqr5` tests after each step of a bulge (vigilant deflation).
    template <typename T>
    void vigilant_deflation(one_based<T>  h,
                            lapack_index  k,
                            lapack_index  ktop,
                            lapack_index  kbot,
                            double        smallest)
    {
      if (h(k + 1, k) == T {}) {
        return;
      }
      double test {modulus_1(h(k, k)) + modulus_1(h(k + 1, k + 1))};
      if (test == 0.0) {
        for (lapack_index d {1}; d <= 3; ++d) {
          if (k >= ktop + d) {
            test += std::abs(real_part(h(k, k - d)));
          }
          if (k <= kbot - d - 1) {
            test += std::abs(real_part(h(k + d + 1, k + 1)));
          }
        }
      }
      if (modulus_1(h(k + 1, k)) > std::max(smallest, schur_ulp * test)) {
        return;
      }
      const double h12   {std::max(modulus_1(h(k + 1, k)), modulus_1(h(k, k + 1)))};
      const double h21   {std::min(modulus_1(h(k + 1, k)), modulus_1(h(k, k + 1)))};
      const double h11   {std::max(modulus_1(h(k + 1, k + 1)), modulus_1(h(k, k) - h(k + 1, k + 1)))};
      const double h22   {std::min(modulus_1(h(k + 1, k + 1)), modulus_1(h(k, k) - h(k + 1, k + 1)))};
      const double scale {h11 + h12};
      const double test2 {h22 * (h11 / scale)};
      if (test2 == 0.0 || h21 * (h12 / scale) <= std::max(smallest, schur_ulp * test2)) {
        h(k + 1, k) = T {};
      }
    }

    //! \brief The adjoint `I − τ̄ v vᴴ` of a reflection of three entries from the left on one column (\p x three entries
    //!        \p stride apart).
    template <typename T>
    void reflect_from_left(T*          x,
                           std::size_t stride,
                           const T&    tau,
                           const T&    v2,
                           const T&    v3)
    {
      const T sum {times(conj_of(tau), x[0] + times(conj_of(v2), x[stride]) + times(conj_of(v3), x[2 * stride]))};
      x[0]          -= sum;
      x[stride]     -= times(sum, v2);
      x[2 * stride] -= times(sum, v3);
    }

    /*!
     * \brief One small-bulge multishift QR sweep over the active block \p ktop … \p kbot of \p target's Hessenberg
     *        matrix with the \p count shifts at \p s, as LAPACK's `dlaqr5` and `zlaqr5`: chains of 3 × 3 bulges, two
     *        shifts each, chased down the diagonal by slabs of `4 · count / 2` columns; with \p accumulate, each
     *        slab's reflections are accumulated in `U` and applied to the rest of the matrix and to `Z` through the matrix
     *        product kernel, else applied to them directly (LAPACK's `kacc22`, which accumulates from 14 shifts). A
     *        subdiagonal entry that becomes negligible on the way is set to zero (vigilant deflation).
     */
    template <typename T>
    void chase_bulges(const schur_target<T>& target,
                      lapack_index           ktop,
                      lapack_index           kbot,
                      std::complex<double>*  s,
                      lapack_index           count,
                      bool                   accumulate)
    {
      const lapack_index n        {static_cast<lapack_index>(target.n)};
      const one_based<T> h        {target.h->data(), n};
      pair_shifts(s, count, T {});
      const lapack_index ns       {count - (count % 2)};
      const double       smallest {std::numeric_limits<double>::min() * (static_cast<double>(n) / schur_ulp)};
      if (ktop + 2 <= kbot) {
        h(ktop + 2, ktop) = T {};
      }
      const lapack_index nbmps {ns / 2};
      const lapack_index kdu {4 * nbmps};
      buffer<T>          u_store(count_of(kdu * kdu));
      const one_based<T> ut {u_store.data(), kdu}; // Uᵀ: the accumulation updates columns of U, contiguous in it
      buffer<T>          v_store(count_of(3 * nbmps), T {});
      const auto         v {[&v_store](lapack_index i, lapack_index m) -> T& { return v_store[count_of(((m - 1) * 3) + (i - 1))]; }};
      const lapack_index low  {static_cast<lapack_index>(target.low) + 1};
      const lapack_index high {static_cast<lapack_index>(target.high)};
      for (lapack_index incol {ktop - (2 * nbmps) + 1}; incol <= kbot - 2; incol += 2 * nbmps) {
        const lapack_index ndcol {incol + kdu};
        const lapack_index jtop {accumulate ? std::max(ktop, incol) : (target.full ? 1 : ktop)};
        const lapack_index jbot {accumulate ? std::min(ndcol, kbot) : (target.full ? n : kbot)};
        const auto         to_z {[&](lapack_index k, lapack_index m, std::size_t entries) { // Z ← Z H, on Zᵀ's rows
                                   if (accumulate || target.zt == nullptr) {
                                     return;
                                   }
                                   const one_based<T> zt {target.zt->data(), n};
                                   if (entries == 3) {
                                     reflect_three_rows(&zt(k + 1, 1), &zt(k + 2, 1), &zt(k + 3, 1), count_of(low - 1), count_of(high), v(1, m), v(2, m), v(3, m), false);
                                   }
                                   else {
                                     reflect_two_rows(&zt(k + 1, 1), &zt(k + 2, 1), count_of(low - 1), count_of(high), v(1, m), v(2, m), false);
                                   }
                                 }};
        std::fill(u_store.begin(), u_store.end(), T {});
        for (lapack_index d {1}; d <= kdu; ++d) {
          ut(d, d) = T {1};
        }
        for (lapack_index krcol {incol}; krcol <= std::min(incol + (2 * nbmps) - 1, kbot - 2); ++krcol) {
          const lapack_index mtop  {std::max<lapack_index>(1, ((ktop - krcol) / 2) + 1)};
          const lapack_index mbot  {std::min(nbmps, (kbot - krcol - 1) / 2)};
          const lapack_index m22   {mbot + 1};
          const bool         bmp22 {mbot < nbmps && krcol + (2 * (m22 - 1)) == kbot - 2};
          if (bmp22) { // a 2 × 2 reflection at the bottom
            const lapack_index k {krcol + (2 * (m22 - 1))};
            if (k == ktop - 1) {
              std::array<T, 2> column {};
              bulge_column(one_based<T> {&h(k + 1, k + 1), n}, 2, s[(2 * m22) - 2], s[(2 * m22) - 1], column.data());
              T beta                  {column[0]};
              v(2, m22) = column[1];
              v(1, m22) = reflector_tau(beta, &v(2, m22), 1);
            }
            else {
              T beta {h(k + 1, k)};
              v(2, m22)    = h(k + 2, k);
              v(1, m22)    = reflector_tau(beta, &v(2, m22), 1);
              h(k + 1, k)  = beta;
              h(k + 2, k)  = T {};
            }
            reflect_two_columns(&h(jtop, k + 1), count_of(std::min(kbot, k + 3) - jtop + 1), count_of(n), v(1, m22), v(2, m22));
            reflect_two_rows(&h(k + 1, 1), &h(k + 2, 1), count_of(k), count_of(jbot), v(1, m22), v(2, m22), true);
            if (k >= ktop) {
              vigilant_deflation(h, k, ktop, kbot, smallest);
            }
            const lapack_index kms {k - incol};
            if (accumulate) {
              const std::size_t from {count_of(std::max<lapack_index>(1, ktop - incol) - 1)};
              reflect_two_rows(&ut(kms + 1, 1), &ut(kms + 2, 1), from, count_of(kdu), v(1, m22), v(2, m22), false);
            }
            to_z(k, m22, 2);
          }
          for (lapack_index m {mbot}; m >= mtop; --m) { // the chain of 3 × 3 reflections
            const lapack_index k {krcol + (2 * (m - 1))};
            if (k == ktop - 1) {
              std::array<T, 3> column {};
              bulge_column(one_based<T> {&h(ktop, ktop), n}, 3, s[(2 * m) - 2], s[(2 * m) - 1], column.data());
              T alpha                 {column[0]};
              v(2, m) = column[1];
              v(3, m) = column[2];
              v(1, m) = reflector_tau(alpha, &v(2, m), 2);
            }
            else {
              // the row below the bulge takes the previous reflection, its first two entries being zero
              const T sum {v(1, m) * v(3, m) * h(k + 3, k + 2)};
              h(k + 3, k)      = -sum;
              h(k + 3, k + 1)  = -sum * conj_of(v(2, m));
              h(k + 3, k + 2) -= sum * conj_of(v(3, m));
              T beta {h(k + 1, k)};
              v(2, m) = h(k + 2, k);
              v(3, m) = h(k + 3, k);
              v(1, m) = reflector_tau(beta, &v(2, m), 2);
              // A bulge collapses (vigilant deflation, or destructive underflow) where the row below it holds only its
              // last entry: then a bulge started afresh from the shifts replaces it where its fill would be negligible
              // (the fill of none is infinite).
              double           fill      {std::numeric_limits<double>::infinity()};
              std::array<T, 3> fresh     {};
              T                fresh_tau {};
              T                fresh_sum {};
              if (h(k + 3, k) == T {} && h(k + 3, k + 2) != T {}) { // the entry between is −sum v̄₂: zero with it
                bulge_column(one_based<T> {&h(k + 1, k + 1), n}, 3, s[(2 * m) - 2], s[(2 * m) - 1], fresh.data());
                T alpha {fresh[0]};
                fresh_tau = reflector_tau(alpha, fresh.data() + 1, 2);
                fresh_sum = conj_of(fresh_tau) * (h(k + 1, k) + (conj_of(fresh[1]) * h(k + 2, k)));
                fill      = modulus_1(h(k + 2, k) - (fresh_sum * fresh[1])) + modulus_1(fresh_sum * fresh[2])
                            - (schur_ulp * (modulus_1(h(k, k)) + modulus_1(h(k + 1, k + 1)) + modulus_1(h(k + 2, k + 2))));
              }
              if (fill <= 0.0) {
                h(k + 1, k) -= fresh_sum;
                v(1, m)      = fresh_tau;
                v(2, m)      = fresh[1];
                v(3, m)      = fresh[2];
              }
              else {
                h(k + 1, k) = beta;
              }
              h(k + 2, k) = T {};
              h(k + 3, k) = T {};
            }
            reflect_three_columns(&h(jtop, k + 1), count_of(std::min(kbot, k + 3) - jtop + 1), count_of(n), v(1, m), v(2, m), v(3, m));
            reflect_from_left(&h(k + 1, k + 1), count_of(n), v(1, m), v(2, m), v(3, m));
            if (k >= ktop) {
              vigilant_deflation(h, k, ktop, kbot, smallest);
            }
          }
          for (lapack_index m {mbot}; m >= mtop; --m) { // the rest of the left reflections, within the slab
            const lapack_index k {krcol + (2 * (m - 1))};
            reflect_three_rows(&h(k + 1, 1), &h(k + 2, 1), &h(k + 3, 1), count_of(std::max(ktop, krcol + (2 * m)) - 1), count_of(jbot), v(1, m),
                               v(2, m), v(3, m), true);
          }
          for (lapack_index m {mbot}; m >= mtop; --m) { // accumulated in U
            const lapack_index k   {krcol + (2 * (m - 1))};
            const lapack_index kms {k - incol};
            const lapack_index i2  {std::max({lapack_index {1}, ktop - incol, kms - (krcol - incol) + 1})};
            const lapack_index i4  {std::min(kdu, krcol + (2 * (mbot - 1)) - incol + 5)};
            if (accumulate) {
              reflect_three_rows(&ut(kms + 1, 1), &ut(kms + 2, 1), &ut(kms + 3, 1), count_of(i2 - 1), count_of(i4), v(1, m), v(2, m), v(3, m), false);
            }
            to_z(k, m, 3);
          }
        }
        if (!accumulate) {
          continue;
        }
        // U to the rest of the matrix: the rows right of the slab, the columns above it, and Z
        const lapack_index jtop_far {target.full ? 1 : ktop};
        const lapack_index jbot_far {target.full ? n : kbot};
        const lapack_index k1       {std::max<lapack_index>(1, ktop - incol)};
        const lapack_index nu       {(kdu - std::max<lapack_index>(0, ndcol - kbot)) - k1 + 1}; // ≥ kbot − ktop + 1
        buffer<T>          u_block(count_of(nu * nu));
        buffer<T>          u_adjoint(count_of(nu * nu));
        for (lapack_index i {1}; i <= nu; ++i) {
          for (lapack_index j {1}; j <= nu; ++j) {
            u_block[count_of(((i - 1) * nu) + (j - 1))]   = ut(k1 + j - 1, k1 + i - 1);
            u_adjoint[count_of(((j - 1) * nu) + (i - 1))] = conj_of(ut(k1 + j - 1, k1 + i - 1));
          }
        }
        const lapack_index first_column {std::min(ndcol, kbot) + 1};
        if (first_column <= jbot_far) {
          const std::size_t width                           {count_of(jbot_far - first_column + 1)};
          buffer<T>         product(count_of(nu) * width, T {});
          gemm(count_of(nu), width, count_of(nu), u_adjoint.data(), count_of(nu), &h(incol + k1, first_column), count_of(n), product.data(), width);
          for (lapack_index i {1}; i <= nu; ++i) {
            std::copy_n(product.data() + (count_of(i - 1) * width), width, &h(incol + k1 + i - 1, first_column));
          }
        }
        const lapack_index last_row {std::max(ktop, incol) - 1};
        if (jtop_far <= last_row) {
          const std::size_t rows                           {count_of(last_row - jtop_far + 1)};
          buffer<T>         product(rows * count_of(nu), T {});
          gemm(rows, count_of(nu), count_of(nu), &h(jtop_far, incol + k1), count_of(n), u_block.data(), count_of(nu), product.data(), count_of(nu));
          for (std::size_t r {0}; r < rows; ++r) {
            std::copy_n(product.data() + (r * count_of(nu)), count_of(nu), &h(jtop_far + static_cast<lapack_index>(r), incol + k1));
          }
        }
        if (target.zt != nullptr) { // (Z U)ᵀ = Uᵀ Zᵀ, on Zᵀ's rows
          buffer<T> u_transpose(count_of(nu * nu));
          for (lapack_index i {1}; i <= nu; ++i) {
            for (lapack_index j {1}; j <= nu; ++j) {
              u_transpose[count_of(((j - 1) * nu) + (i - 1))] = ut(k1 + j - 1, k1 + i - 1);
            }
          }
          const one_based<T> zt                              {target.zt->data(), n};
          const std::size_t  width                           {count_of(high - low + 1)};
          buffer<T>          product(count_of(nu) * width, T {});
          gemm(count_of(nu), width, count_of(nu), u_transpose.data(), count_of(nu), &zt(incol + k1, low), count_of(n), product.data(), width);
          for (lapack_index i {1}; i <= nu; ++i) {
            std::copy_n(product.data() + (count_of(i - 1) * width), width, &zt(incol + k1 + i - 1, low));
          }
        }
      }
    }

    template <typename T>
    void multishift_schur(const schur_target<T>& target,
                          std::complex<double>*  w,
                          bool                   recursive,
                          lapack_index           steps = 0);

    //! \brief The Schur form of \p t's active block by \ref real_schur, its eigenvalues to \p w (one per row of \p t).
    inline void plain_schur(const schur_target<double>& t,
                            std::complex<double>*       w)
    {
      buffer<double> re(t.n, 0.0);
      buffer<double> im(t.n, 0.0);
      real_schur(t, re, im, schur_steps_per_order * std::max<std::size_t>(10, t.high - t.low));
      for (std::size_t i {t.low}; i < t.high; ++i) {
        w[i] = std::complex<double> {re[i], im[i]};
      }
    }

    //! \brief The same by \ref complex_schur.
    inline void plain_schur(const schur_target<std::complex<double>>& t,
                            std::complex<double>*                     w)
    {
      buffer<std::complex<double>> values(t.n);
      complex_schur(t, values, schur_steps_per_order * std::max<std::size_t>(10, t.high - t.low));
      std::copy(values.begin() + static_cast<std::ptrdiff_t>(t.low), values.begin() + static_cast<std::ptrdiff_t>(t.high), w + t.low);
    }

    /*!
     * \brief The eigenvalues of the real Schur form \p t (`jw × jw`) of a deflation window that do not deflate, moved
     *        to its top, as `dlaqr3`: from the bottom, a block whose spike entries `s v₁ⱼ` are negligible deflates, any
     *        other is moved up; the undeflated blocks are then sorted by decreasing modulus (which helps graded
     *        matrices), and every eigenvalue of the window written to \p w. \return Their count.
     */
    inline lapack_index undeflated(one_based<double>     t,
                                   one_based<double>     vt,
                                   lapack_index          jw,
                                   double                s,
                                   double                smallest,
                                   std::complex<double>* w)
    {
      const auto   size {[&t](lapack_index i, lapack_index k) {
                           return k == i + 1 ? std::abs(t(i, i)) : std::abs(t(i, i)) + (std::sqrt(std::abs(t(i + 1, i))) * std::sqrt(std::abs(t(i, i + 1))));
                         }};
      lapack_index ns   {jw};
      lapack_index ilst {1};
      while (ilst <= ns) {
        const lapack_index order {ns > 1 && t(ns, ns - 1) != 0.0 ? 2 : 1};
        double             foo   {order == 1 ? std::abs(t(ns, ns)) : size(ns - 1, ns + 1)};
        if (foo == 0.0) {
          foo = std::abs(s);
        }
        const double spike {order == 1 ? std::abs(s * vt(ns, 1)) : std::max(std::abs(s * vt(ns, 1)), std::abs(s * vt(ns - 1, 1)))};
        if (spike <= std::max(smallest, schur_ulp * foo)) {
          ns -= order;
        }
        else {
          (void)reorder(t, vt, jw, ns, ilst);
          ilst += order;
        }
      }
      if (ns < jw) { // bubble sort of the undeflated blocks, larger first
        bool         sorted {false};
        lapack_index i {ns + 1};
        const auto   next {[&t](lapack_index from, lapack_index end) { return from == end || t(from + 1, from) == 0.0 ? from + 1 : from + 2; }};
        while (!sorted) {
          sorted = true;
          const lapack_index kend {i - 1};
          i = 1;
          lapack_index k          {next(i, ns)};
          while (k <= kend) {
            const double evi {size(i, k)};
            const double evk {k == kend ? std::abs(t(k, k)) : size(k, next(k, kend))};
            if (evi >= evk) {
              i = k;
            }
            else {
              sorted = false;
              lapack_index last {k};
              i = reorder(t, vt, jw, i, last) ? last : k;
            }
            k = next(i, kend);
          }
        }
      }
      for (lapack_index i {jw}; i >= 1;) {
        if (i == 1 || t(i, i - 1) == 0.0) {
          w[i - 1] = std::complex<double> {t(i, i), 0.0};
          --i;
        }
        else {
          double               a     {t(i - 1, i - 1)};
          double               b     {t(i - 1, i)};
          double               c     {t(i, i - 1)};
          double               d     {t(i, i)};
          const standard_block block {standardize_block(a, b, c, d)};
          w[i - 2]  = std::complex<double> {block.re1, block.im1};
          w[i - 1]  = std::complex<double> {block.re2, block.im2};
          i        -= 2;
        }
      }
      return ns;
    }

    //! \brief The same for a complex Schur form, as `zlaqr3`: 1 × 1 blocks only, sorted by selection.
    inline lapack_index undeflated(one_based<std::complex<double>> t,
                                   one_based<std::complex<double>> vt,
                                   lapack_index                    jw,
                                   const std::complex<double>&     s,
                                   double                          smallest,
                                   std::complex<double>*           w)
    {
      lapack_index ns   {jw};
      lapack_index ilst {1};
      for (lapack_index count {1}; count <= jw; ++count) {
        double foo {modulus_1(t(ns, ns))};
        if (foo == 0.0) {
          foo = modulus_1(s);
        }
        if (modulus_1(s) * modulus_1(vt(ns, 1)) <= std::max(smallest, schur_ulp * foo)) {
          --ns;
        }
        else {
          (void)reorder(t, vt, jw, ns, ilst);
          ++ilst;
        }
      }
      if (ns < jw) {
        for (lapack_index i {1}; i <= ns; ++i) {
          lapack_index first {i};
          for (lapack_index j {i + 1}; j <= ns; ++j) {
            first = modulus_1(t(j, j)) > modulus_1(t(first, first)) ? j : first;
          }
          lapack_index last {i};
          (void)reorder(t, vt, jw, first, last);
        }
      }
      for (lapack_index i {1}; i <= jw; ++i) {
        w[i - 1] = t(i, i);
      }
      return ns;
    }

    //! \brief What \ref early_deflation found: the eigenvalues left as shifts, and those deflated.
    struct deflation_count
    {
      lapack_index shifts   {0}; //!< Undeflated: the shifts, at the bottom of the active block.
      lapack_index deflated {0}; //!< Deflated below them.
    };

    /*!
     * \brief Aggressive early deflation, as LAPACK's `dlaqr3` and `zlaqr3`: the trailing \p nw × \p nw window of the
     *        active block \p ktop … \p kbot is brought to Schur form; its eigenvalues whose coupling to the rest (the
     *        spike, `s` times the first row of the window's Schur vectors) is negligible deflate; the window is reduced
     *        back to Hessenberg form above them, and its transformation carried to the rest of `H` and to `Z` through
     *        the matrix product kernel. Every eigenvalue of the window is written to \p w: the deflated ones final, the
     *        others the next shifts. \p recursive: the window itself by \ref multishift_schur past its crossover.
     */
    template <typename T>
    [[nodiscard]] deflation_count early_deflation(const schur_target<T>& target,
                                                  lapack_index           ktop,
                                                  lapack_index           kbot,
                                                  lapack_index           nw,
                                                  std::complex<double>*  w,
                                                  bool                   recursive)
    {
      const lapack_index n        {static_cast<lapack_index>(target.n)};
      const one_based<T> h        {target.h->data(), n};
      const double       smallest {std::numeric_limits<double>::min() * (static_cast<double>(n) / schur_ulp)};
      const lapack_index jw       {std::min(nw, kbot - ktop + 1)};
      const lapack_index kwtop    {kbot - jw + 1};
      T                  s        {kwtop == ktop ? T {} : h(kwtop, kwtop - 1)};
      if (kbot == kwtop) { // a 1 × 1 window: the whole active block (a window has two rows otherwise), so s = 0
        w[kwtop - 1] = h(kwtop, kwtop);
        return {0, 1};
      }
      const std::size_t  width                     {count_of(jw)};
      buffer<T>          t_store(width * width, T  {});
      buffer<T>          vt_store(width * width, T {});
      const one_based<T> t                         {t_store.data(), jw};
      const one_based<T> vt                        {vt_store.data(), jw};
      for (lapack_index i {1}; i <= jw; ++i) {
        for (lapack_index j {std::max<lapack_index>(1, i - 1)}; j <= jw; ++j) {
          t(i, j) = h(kwtop + i - 1, kwtop + j - 1);
        }
        vt(i, i) = T {1};
      }
      buffer<std::complex<double>> local(width);
      const schur_target<T>        window {&t_store, width, 0, width, true, &vt_store};
      if (recursive && jw >= multishift_from(T {})) {
        multishift_schur(window, local.data(), false);
      }
      else {
        plain_schur(window, local.data());
      }
      for (lapack_index j {1}; j + 3 <= jw; ++j) { // the exchanges need a clean margin below the diagonal
        t(j + 2, j) = T {};
        t(j + 3, j) = T {};
      }
      if (jw > 2) {
        t(jw, jw - 2) = T {};
      }
      const lapack_index ns {undeflated(t, vt, jw, s, smallest, w + kwtop - 1)};
      if (ns == 0) {
        s = T {};
      }
      if (ns == jw) { // nothing deflated (s ≠ 0 then: s = 0 deflates everything): H as it was
        return {ns, 0};
      }
      if (ns > 1) {   // the spike reflected back into the first column, then Hessenberg form above
        buffer<T> v(count_of(ns));
        for (lapack_index i {1}; i <= ns; ++i) {
          v[count_of(i - 1)] = conj_of(vt(i, 1));
        }
        T       beta {v[0]};
        const T tau  {reflector_tau(beta, v.data() + 1, count_of(ns - 1))};
        v[0] = T {1};
        for (lapack_index i {3}; i <= jw; ++i) {
          for (lapack_index j {1}; j + 2 <= i; ++j) {
            t(i, j) = T {};
          }
        }
        for (lapack_index c {1}; c <= jw; ++c) { // (I − τ̄ v vᴴ) on rows 1 … ns
          T sum {};
          for (lapack_index r {1}; r <= ns; ++r) {
            sum += times(conj_of(v[count_of(r - 1)]), t(r, c));
          }
          for (lapack_index r {1}; r <= ns; ++r) {
            t(r, c) -= times(times(conj_of(tau), v[count_of(r - 1)]), sum);
          }
        }
        for (lapack_index r {1}; r <= ns; ++r) { // (I − τ v vᴴ) on columns 1 … ns
          T sum {};
          for (lapack_index c {1}; c <= ns; ++c) {
            sum += times(t(r, c), v[count_of(c - 1)]);
          }
          for (lapack_index c {1}; c <= ns; ++c) {
            t(r, c) -= times(times(tau, sum), conj_of(v[count_of(c - 1)]));
          }
        }
        for (lapack_index c {1}; c <= jw; ++c) { // V ← V (I − τ v vᴴ): Vᵀ's rows 1 … ns
          T sum {};
          for (lapack_index r {1}; r <= ns; ++r) {
            sum += times(v[count_of(r - 1)], vt(r, c));
          }
          for (lapack_index r {1}; r <= ns; ++r) {
            vt(r, c) -= times(times(tau, conj_of(v[count_of(r - 1)])), sum);
          }
        }
        const buffer<T> qt                           {hessenberg(t_store, width, 0, count_of(ns), true)};
        buffer<T>       rows(count_of(ns) * width, T {});
        gemm(count_of(ns), width, count_of(ns), qt.data(), width, vt_store.data(), width, rows.data(), width);
        std::copy(rows.begin(), rows.end(), vt_store.begin());
      }
      if (kwtop > 1) {
        h(kwtop, kwtop - 1) = s * conj_of(vt(1, 1));
      }
      for (lapack_index i {1}; i <= jw; ++i) {
        for (lapack_index j {std::max<lapack_index>(1, i - 1)}; j <= jw; ++j) {
          h(kwtop + i - 1, kwtop + j - 1) = t(i, j);
        }
      }
      buffer<T> v(width * width);
      buffer<T> v_adjoint(width * width);
      for (lapack_index i {1}; i <= jw; ++i) {
        for (lapack_index j {1}; j <= jw; ++j) {
          v[count_of(((j - 1) * jw) + (i - 1))]         = vt(i, j);
          v_adjoint[count_of(((i - 1) * jw) + (j - 1))] = conj_of(vt(i, j));
        }
      }
      const lapack_index ltop {target.full ? 1 : ktop};
      if (ltop < kwtop) { // the rows above the window: H V
        const std::size_t rows                    {count_of(kwtop - ltop)};
        buffer<T>         product(rows * width, T {});
        gemm(rows, width, width, &h(ltop, kwtop), count_of(n), v.data(), width, product.data(), width);
        for (std::size_t r {0}; r < rows; ++r) {
          std::copy_n(product.data() + (r * width), width, &h(ltop + static_cast<lapack_index>(r), kwtop));
        }
      }
      if (target.full && kbot < n) { // the columns right of it: Vᴴ H
        const std::size_t columns                    {count_of(n - kbot)};
        buffer<T>         product(width * columns, T {});
        gemm(width, columns, width, v_adjoint.data(), width, &h(kwtop, kbot + 1), count_of(n), product.data(), columns);
        for (std::size_t r {0}; r < width; ++r) {
          std::copy_n(product.data() + (r * columns), columns, &h(kwtop + static_cast<lapack_index>(r), kbot + 1));
        }
      }
      if (target.zt != nullptr) { // (Z V)ᵀ = Vᵀ Zᵀ
        const one_based<T> zt                         {target.zt->data(), n};
        const lapack_index low                        {static_cast<lapack_index>(target.low) + 1};
        const std::size_t  columns                    {target.high - target.low};
        buffer<T>          product(width * columns, T {});
        gemm(width, columns, width, vt_store.data(), width, &zt(kwtop, low), count_of(n), product.data(), columns);
        for (std::size_t r {0}; r < width; ++r) {
          std::copy_n(product.data() + (r * columns), columns, &zt(kwtop + static_cast<lapack_index>(r), low));
        }
      }
      return {ns, jw - ns};
    }

    //! \brief The shifts LAPACK's `iparmq` recommends for an active block of order \p nh.
    [[nodiscard]] inline lapack_index recommended_shifts(lapack_index nh)
    {
      lapack_index ns {2};
      if (nh >= 6000) {
        ns = 256;
      }
      else if (nh >= 3000) {
        ns = 128;
      }
      else if (nh >= 590) {
        ns = 64;
      }
      else if (nh >= 150) {
        ns = std::max<lapack_index>(10, nh / static_cast<lapack_index>(std::lround(std::log2(static_cast<double>(nh)))));
      }
      else if (nh >= 60) {
        ns = 10;
      }
      else if (nh >= 30) {
        ns = 4;
      }
      return std::max<lapack_index>(2, ns - (ns % 2));
    }

    //! \brief The deflation window LAPACK's `iparmq` recommends for an active block of order \p nh and \p ns shifts.
    [[nodiscard]] inline lapack_index recommended_window(lapack_index nh,
                                                         lapack_index ns)
    {
      return nh <= 500 ? ns : 3 * ns / 2;
    }

    //! \brief Exceptional shifts for a real matrix, as `dlaqr0`'s after `kexsh` iterations without deflation: from the
    //!        subdiagonal entries, as conjugate pairs of the 2 × 2 blocks `dlahqr`'s exceptional shifts build.
    inline void exceptional_shifts(one_based<double>     h,
                                   std::complex<double>* w,
                                   lapack_index          ks,
                                   lapack_index          kbot,
                                   lapack_index          ktop)
    {
      for (lapack_index i {kbot}; i >= std::max(ks + 1, ktop + 2); i -= 2) {
        const double         ss    {std::abs(h(i, i - 1)) + std::abs(h(i - 1, i - 2))};
        double               a     {(0.75 * ss) + h(i, i)};
        double               b     {ss};
        double               c     {-0.4375 * ss};
        double               d     {a};
        const standard_block block {standardize_block(a, b, c, d)};
        w[i - 2] = std::complex<double> {block.re1, block.im1};
        w[i - 1] = std::complex<double> {block.re2, block.im2};
      }
      // LAPACK's case ks = ktop cannot arise: a sweep's active block has three rows at least (one of two deflates whole
      // in its window), and ns ≤ kbot − ktop.
    }

    //! \brief Exceptional shifts for a complex matrix, as `zlaqr0`'s.
    inline void exceptional_shifts(one_based<std::complex<double>> h,
                                   std::complex<double>*           w,
                                   lapack_index                    ks,
                                   lapack_index                    kbot,
                                   lapack_index /*ktop*/)
    {
      for (lapack_index i {kbot}; i >= ks + 1; i -= 2) {
        w[i - 1] = h(i, i) + (0.75 * modulus_1(h(i, i - 1)));
        w[i - 2] = w[i - 1];
      }
    }

    //! \brief A real matrix's shifts brought into pairs, each a conjugate pair or two reals, as `dlaqr0` does before
    //!        its sweep (conjugate shifts already adjacent).
    inline void pair_for_sweep(std::complex<double>* w,
                               lapack_index          ks,
                               lapack_index          kbot,
                               double /*real*/)
    {
      for (lapack_index i {kbot}; i >= ks + 2; i -= 2) {
        if (w[i - 1].imag() != -w[i - 2].imag()) {
          const std::complex<double> last {w[i - 1]};
          w[i - 1] = w[i - 2];
          w[i - 2] = w[i - 3];
          w[i - 3] = last;
        }
      }
    }

    //! \brief A complex matrix's shifts need no pairing.
    inline void pair_for_sweep(std::complex<double>* /*w*/,
                               lapack_index /*ks*/,
                               lapack_index /*kbot*/,
                               const std::complex<double>& /*complex*/)
    {
    }

    //! \brief Whether a real matrix's sweep is down to two shifts, both real: then the one nearer the corner serves twice.
    [[nodiscard]] inline bool two_real_shifts(const std::complex<double>* w,
                                              lapack_index                ks,
                                              lapack_index                kbot,
                                              double /*real*/)
    {
      return kbot - ks + 1 == 2 && w[kbot - 1].imag() == 0.0;
    }

    //! \brief The same for a complex matrix: two shifts of any kind.
    [[nodiscard]] inline bool two_real_shifts(const std::complex<double>* /*w*/,
                                              lapack_index                ks,
                                              lapack_index                kbot,
                                              const std::complex<double>& /*complex*/)
    {
      return kbot - ks + 1 == 2;
    }

    /*!
     * \brief The Schur form of \p target's active block by the multishift QR algorithm with aggressive early
     *        deflation, as LAPACK's `dlaqr0` and `zlaqr0` (`dlaqr4` without \p recursive): each iteration deflates what
     *        it can from a window at the bottom of the active block (\ref early_deflation) and, unless that deflated
     *        enough, sweeps the block with the window's other eigenvalues as shifts (\ref chase_bulges). The eigenvalues
     *        are left in \p w (one per row; a real matrix's complex pairs adjacent, the positive imaginary part first).
     * \param[in]  target    The Hessenberg matrix, its active block, and the Schur vectors to update.
     * \param[out] w         The eigenvalues, one per row of the matrix.
     * \param[in]  recursive Whether a wide deflation window is itself taken by these iterations (`dlaqr0`), or by the
     *                       single-shift ones (`dlaqr4`).
     * \param[in]  steps     The iterations allowed, LAPACK's `30 · max(10, order)` where zero.
     * \throws convergence_error If the iterations did not converge within \p steps.
     */
    template <typename T>
    void multishift_schur(const schur_target<T>& target,
                          std::complex<double>*  w,
                          bool                   recursive,
                          lapack_index           steps)
    {
      const lapack_index n           {static_cast<lapack_index>(target.n)};
      const one_based<T> h           {target.h->data(), n};
      const lapack_index ilo         {static_cast<lapack_index>(target.low) + 1};
      const lapack_index ihi         {static_cast<lapack_index>(target.high)};
      const lapack_index nh          {ihi - ilo + 1};
      const lapack_index recommended {recommended_shifts(nh)};
      const bool         accumulate  {recommended >= 14};
      const lapack_index nwr         {std::min({nh, (n - 1) / 3, std::max<lapack_index>(2, recommended_window(nh, recommended))})};
      lapack_index       nsr         {std::min({recommended, (n - 3) / 6, ihi - ilo})};
      nsr = std::max<lapack_index>(2, nsr - (nsr % 2));
      const lapack_index nwmax       {(n - 1) / 3};
      const lapack_index nsmax       {((n - 3) / 6) - (((n - 3) / 6) % 2)};
      lapack_index       nw          {nwmax};
      lapack_index       ndfl        {1};
      lapack_index       ndec        {-1};
      lapack_index       kbot        {ihi};
      const lapack_index itmax       {steps > 0 ? steps : 30 * std::max<lapack_index>(10, nh)};
      for (lapack_index it {1}; it <= itmax; ++it) {
        if (kbot < ilo) {
          return;
        }
        lapack_index ktop {kbot};
        while (ktop > ilo && h(ktop, ktop - 1) != T {}) {
          --ktop;
        }
        const lapack_index active {kbot - ktop + 1};
        const lapack_index nwupbd {std::min(active, nwmax)};
        nw = ndfl < window_growth ? std::min(nwupbd, nwr) : std::min(nwupbd, 2 * nw);
        if (nw < nwmax) {
          if (nw >= active - 1) {
            nw = active;
          }
          else if (modulus_1(h(kbot - nw + 1, kbot - nw)) > modulus_1(h(kbot - nw, kbot - nw - 1))) {
            ++nw;
          }
        }
        if (ndfl < window_growth) {
          ndec = -1;
        }
        else if (ndec >= 0 || nw >= nwupbd) {
          ++ndec;
          if (nw - ndec < 2) {
            ndec = 0;
          }
          nw -= ndec;
        }
        const deflation_count found {early_deflation(target, ktop, kbot, nw, w, recursive)};
        kbot -= found.deflated;
        lapack_index ks             {kbot - found.shifts + 1};
        if (found.deflated == 0 || (100 * found.deflated <= nw * nibble && kbot - ktop + 1 > std::min(multishift_crossover, nwmax))) {
          lapack_index ns {std::min({nsmax, nsr, std::max<lapack_index>(2, kbot - ktop)})};
          ns -= ns % 2;
          if (ndfl % exceptional_after == 0) {
            ks = kbot - ns + 1;
            exceptional_shifts(h, w, ks, kbot, ktop);
          }
          else {
            if (kbot - ks + 1 <= ns / 2) { // too few: the eigenvalues of the trailing ns × ns
              ks = kbot - ns + 1;
              buffer<T>          trailing(count_of(ns * ns), T {});
              const one_based<T> lower                         {trailing.data(), ns};
              for (lapack_index i {1}; i <= ns; ++i) {
                for (lapack_index j {std::max<lapack_index>(1, i - 1)}; j <= ns; ++j) {
                  lower(i, j) = h(ks + i - 1, ks + j - 1);
                }
              }
              // by the single-shift iterations: LAPACK's dlaqr4 serves only past order 3000, where ns exceeds 75
              plain_schur(schur_target<T> {&trailing, count_of(ns), 0, count_of(ns), false, nullptr}, w + ks - 1);
            }
            if (kbot - ks + 1 > ns) { // bubble sort, larger first: a real matrix's pairs stay together
              bool sorted {false};
              for (lapack_index k {kbot}; k > ks && !sorted; --k) {
                sorted = true;
                for (lapack_index i {ks}; i < k; ++i) {
                  if (modulus_1(w[i - 1]) < modulus_1(w[i])) {
                    std::swap(w[i - 1], w[i]);
                    sorted = false;
                  }
                }
              }
            }
            pair_for_sweep(w, ks, kbot, T {});
          }
          if (two_real_shifts(w, ks, kbot, T {})) {
            // two shifts, real for a real matrix: the one nearer the corner, twice
            const T corner {h(kbot, kbot)};
            if (modulus_1(as_scalar(w[kbot - 1], T {}) - corner) < modulus_1(as_scalar(w[kbot - 2], T {}) - corner)) {
              w[kbot - 2] = w[kbot - 1];
            }
            else {
              w[kbot - 1] = w[kbot - 2];
            }
          }
          ns  = std::min(ns, kbot - ks + 1);
          ns -= ns % 2;
          ks  = kbot - ns + 1;
          chase_bulges(target, ktop, kbot, w + ks - 1, ns, accumulate);
        }
        ndfl = found.deflated > 0 ? 1 : ndfl + 1;
      }
      throw convergence_error("eigendecomposition: the QR iterations did not converge");
    }

    //! \brief The real Schur form of \p t's active block: \ref real_schur below \ref multishift_from, else
    //!        \ref multishift_schur, as LAPACK's `dhseqr` chooses. \p steps bounds \ref real_schur's iterations.
    inline void schur_form(const schur_target<double>& t,
                           buffer<double>&             re,
                           buffer<double>&             im,
                           std::size_t                 steps)
    {
      if (t.high - t.low < count_of(multishift_from(0.0))) {
        real_schur(t, re, im, steps);
        return;
      }
      buffer<std::complex<double>> w(t.n);
      multishift_schur(t, w.data(), true);
      for (std::size_t i {t.low}; i < t.high; ++i) {
        re[i] = w[i].real();
        im[i] = w[i].imag();
      }
    }

    //! \brief The complex Schur form likewise: \ref complex_schur or \ref multishift_schur.
    inline void schur_form(const schur_target<std::complex<double>>& t,
                           buffer<std::complex<double>>&             values,
                           std::size_t                               steps)
    {
      if (t.high - t.low < count_of(multishift_from(std::complex<double> {}))) {
        complex_schur(t, values, steps);
        return;
      }
      multishift_schur(t, values.data(), true);
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

    //! \brief A quotient of \ref divided and the factor its dividend was scaled by: a type of its own, where a
    //!        `std::pair` of them made GCC on arm64 print, at every caller, a note on its ABI change of GCC 10.1.
    template <typename S>
    struct scaled_quotient
    {
      S      value;  //!< The quotient.
      double factor; //!< The dividend's scaling, 1 where none was needed.
    };

    //! \brief `r / pivot`, a pivot below \p floor taken as \p floor, \p r scaled first where the quotient would pass
    //!        \ref back_substitution_limit. \return The quotient and the factor (1 where none was needed).
    template <typename S, typename P>
    [[nodiscard]] scaled_quotient<S> divided(S      r,
                                             P      pivot,
                                             double floor)
    {
      const double size    {modulus_1(pivot) < floor ? floor : modulus_1(pivot)};
      const P      divisor {modulus_1(pivot) < floor ? P {floor} : pivot};
      double       factor  {1.0};
      if (modulus_1(r) > back_substitution_limit * size) {
        factor  = back_substitution_limit * size / (2.0 * modulus_1(r));
        r      *= factor;
      }
      return {r / divisor, factor};
    }

    //! \brief A real eigenvector during its back-substitution, the sums of rows against it by \ref dot_product.
    struct real_components
    {
      using scalar = double; //!< Its components.

      buffer<double> x;      //!< The components.

      //! \brief `Σ_{l = from…to} row[l] x[l]`.
      [[nodiscard]] double dot(const double* row,
                               std::size_t   from,
                               std::size_t   to) const
      {
        return dot_product(row + from, x.data() + from, to - from + 1);
      }

      //! \brief Sets component \p j to \p value.
      void set(std::size_t j,
               double      value)
      {
        x[j] = value;
      }

      //! \brief Scales the components \p from … \p to by \p factor.
      void scale(std::size_t from,
                 std::size_t to,
                 double      factor)
      {
        for (std::size_t l {from}; l <= to; ++l) {
          x[l] *= factor;
        }
      }
    };

    //! \brief A real matrix's complex eigenvector during its back-substitution, its real and imaginary parts apart: a row
    //!        of the real Schur form against it is two \ref dot_product.
    struct split_components
    {
      using scalar = std::complex<double>; //!< Its components.

      buffer<double> re;                   //!< The real parts.
      buffer<double> im;                   //!< The imaginary parts.

      //! \brief `Σ_{l = from…to} row[l] x[l]`.
      [[nodiscard]] std::complex<double> dot(const double* row,
                                             std::size_t   from,
                                             std::size_t   to) const
      {
        return {dot_product(row + from, re.data() + from, to - from + 1), dot_product(row + from, im.data() + from, to - from + 1)};
      }

      //! \brief Sets component \p j to \p value.
      void set(std::size_t                 j,
               const std::complex<double>& value)
      {
        re[j] = value.real();
        im[j] = value.imag();
      }

      //! \brief Scales the components \p from … \p to by \p factor.
      void scale(std::size_t from,
                 std::size_t to,
                 double      factor)
      {
        for (std::size_t l {from}; l <= to; ++l) {
          re[l] *= factor;
          im[l] *= factor;
        }
      }
    };

    //! \brief A complex matrix's eigenvector during its back-substitution, kept as `(re, −im)` and `(im, re)` pairs: a
    //!        complex row, read as its parts in order ([complex.numbers]), against each is the real or the imaginary part
    //!        of their product, one \ref dot_product each.
    struct interleaved_components
    {
      using scalar = std::complex<double>; //!< Its components.

      buffer<double> real_side;            //!< `(re, −im)` per component: a row against it gives the product's real part.
      buffer<double> imaginary_side;       //!< `(im, re)` per component: its imaginary part.

      //! \brief `Σ_{l = from…to} row[l] x[l]`.
      [[nodiscard]] std::complex<double> dot(const std::complex<double>* row,
                                             std::size_t                 from,
                                             std::size_t                 to) const
      {
        const auto* const parts {reinterpret_cast<const double*>(row)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        const std::size_t count {2 * (to - from + 1)};
        return {dot_product(parts + (2 * from), real_side.data() + (2 * from), count),
                dot_product(parts + (2 * from), imaginary_side.data() + (2 * from), count)};
      }

      //! \brief Sets component \p j to \p value.
      void set(std::size_t                 j,
               const std::complex<double>& value)
      {
        real_side[2 * j]            = value.real();
        real_side[(2 * j) + 1]      = -value.imag();
        imaginary_side[2 * j]       = value.imag();
        imaginary_side[(2 * j) + 1] = value.real();
      }

      //! \brief Component \p j.
      [[nodiscard]] std::complex<double> get(std::size_t j) const
      {
        return {real_side[2 * j], imaginary_side[2 * j]};
      }

      //! \brief Scales the components \p from … \p to by \p factor.
      void scale(std::size_t from,
                 std::size_t to,
                 double      factor)
      {
        for (std::size_t l {2 * from}; l <= (2 * to) + 1; ++l) {
          real_side[l]      *= factor;
          imaginary_side[l] *= factor;
        }
      }
    };

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
        v[r] = times(v[r], turn);
      }
      v[largest] = std::complex<double> {v[largest].real(), 0.0};
    }

    /*!
     * \brief Solves `(T − λ I) x = 0` upward from the component \p k, rows \p k − 1 … 0 of the row-major quasi-triangular
     *        \p t (its 2 × 2 blocks where a subdiagonal entry is nonzero), \p x `[k … top]` given: as LAPACK's `dtrevc`
     *        and `ztrevc`, a pivot below \p floor taken as \p floor, the components scaled down where they would grow
     *        past \ref back_substitution_limit. The sums of rows against \p x are \p X 's, vectorised.
     */
    template <typename T, typename X>
    void solve_upward(const buffer<T>&     t,
                      std::size_t          n,
                      std::size_t          k,
                      std::size_t          top,
                      typename X::scalar   lambda,
                      double               floor,
                      X&                   x)
    {
      using S = typename X::scalar;
      for (std::size_t j {k}; j > 0;) {
        if constexpr (std::is_floating_point_v<T>) {
          if (j >= 2 && t[((j - 1) * n) + j - 2] != 0.0) { // a 2 × 2 block, rows j − 2 and j − 1
            const S      m[4]   {S {t[((j - 2) * n) + j - 2]} - lambda, S {t[((j - 2) * n) + j - 1]}, S {t[((j - 1) * n) + j - 2]},
                                 S {t[((j - 1) * n) + j - 1]} - lambda};
            S            r[2]   {-x.dot(t.data() + ((j - 2) * n), j, top), -x.dot(t.data() + ((j - 1) * n), j, top)};
            const double factor {solve_block(m, r, floor)};
            if (factor != 1.0) {
              x.scale(j, top, factor);
            }
            x.set(j - 2, r[0]);
            x.set(j - 1, r[1]);
            j -= 2;
            continue;
          }
        }
        const auto [value, factor] {divided(-x.dot(t.data() + ((j - 1) * n), j, top), S {t[((j - 1) * n) + j - 1]} - lambda, floor)};
        if (factor != 1.0) {
          x.scale(j, top, factor);
        }
        x.set(j - 1, value);
        --j;
      }
    }

    //! \brief A triangular solve's work for \ref parallel_for over the \p n eigenvectors: `k²/2` products for vector `k`,
    //!        `n²/6` on average.
    [[nodiscard]] inline std::size_t vector_work(std::size_t n) noexcept
    {
      return std::max<std::size_t>(1, (n * n) / 6);
    }

    //! \brief The eigenvector \p task of \p n solves, the short ones and the long ones in turn (0, n − 1, 1, n − 2, …):
    //!        each of \ref parallel_for's runs of tasks then as long as another.
    [[nodiscard]] inline std::size_t balanced_index(std::size_t task,
                                                    std::size_t n) noexcept
    {
      return task % 2 == 0 ? task / 2 : n - 1 - (task / 2);
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
      parallel_for(n, vector_work(n), [&](std::size_t task) {
                     const std::size_t      k {balanced_index(task, n)};
                     interleaved_components x {buffer<double>(2 * (k + 1), 0.0), buffer<double>(2 * (k + 1), 0.0)};
                     const C                lambda {t[(k * n) + k]};
                     x.set(k, C {1.0});
                     solve_upward(t, n, k, k, lambda, std::max(schur_ulp * modulus_1(lambda), smallest), x);
                     for (std::size_t l {0}; l <= k; ++l) {
                       xt[(k * n) + l] = x.get(l);
                     }
                   });
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
      parallel_for(n, vector_work(n), [&](std::size_t task) {
                     const std::size_t k      {balanced_index(task, n)};
                     const double      floor  {std::max(schur_ulp * (std::abs(re[k]) + std::abs(im[k])), smallest)};
                     double*           row_re {out.re.data() + (k * n)};
                     double*           row_im {out.im.data() + (k * n)};
                     if (im[k] == 0.0) {
                       real_components x {buffer<double>(k + 1, 0.0)};
                       x.set(k, 1.0);
                       solve_upward(t, n, k, k, re[k], floor, x);
                       std::copy(x.x.begin(), x.x.begin() + static_cast<std::ptrdiff_t>(k + 1), row_re);
                     }
                     else if (im[k] > 0.0) { // the pair's block is rows k, k + 1; the second's row written here
                       split_components y {buffer<double>(k + 2, 0.0), buffer<double>(k + 2, 0.0)};
                       const double     b {t[(k * n) + k + 1]};
                       const double     c {t[((k + 1) * n) + k]};
                       y.set(k, std::abs(b) >= std::abs(c) ? C {1.0} : C {-im[k] / c});
                       y.set(k + 1, std::abs(b) >= std::abs(c) ? C {0.0, im[k] / b} : C {0.0, 1.0});
                       solve_upward(t, n, k, k + 1, C {re[k], im[k]}, floor, y);
                       for (std::size_t l {0}; l <= k + 1; ++l) {
                         row_re[l]     = y.re[l];
                         row_im[l]     = y.im[l];
                         row_re[l + n] = y.re[l];
                         row_im[l + n] = -y.im[l];
                       }
                     }
                   });
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
      double largest {0.0}; // the largest part of an entry: within √2 of its modulus, and no hypot per entry
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < n_; ++j) {
          largest = std::max(largest, detail::largest_part(input(i, j)));
        }
      }
      // A power of two near the largest magnitude: dividing by it is exact, and keeps the iterations' sums of squares
      // and the back-substitution's growth in range.
      const double   scale        {largest == 0.0 ? 1.0 : std::ldexp(1.0, std::ilogb(largest))};
      buffer<T>      a(n_ * n_, T {});
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < n_; ++j) {
          a[(i * n_) + j] = input(i, j) / detail::from_carried<T>(scale);
        }
      }
      const detail::balancing       balanced {detail::balance(a, n_)};
      buffer<T>                     z {detail::hessenberg(a, n_, balanced.low, balanced.high, vectors)};
      const std::size_t             steps {detail::schur_steps_per_order * std::max<std::size_t>(10, balanced.high - balanced.low)};
      // The whole Schur form past the multishift crossover even for the eigenvalues alone: the sweeps' products then
      // have the same shapes as with the vectors, and the eigenvalues the same bits (2–4 % of the iterations from order
      // 256 on); below it the single-shift iterations give them either way.
      const bool                    whole  {vectors || balanced.high - balanced.low >= detail::count_of(detail::multishift_from(T {}))};
      const detail::schur_target<T> target {&a, n_, balanced.low, balanced.high, whole, vectors ? &z : nullptr}; // z is Zᵀ
      buffer<std::complex<double>>  values(n_);
      buffer<std::complex<double>>  vt;                                                                          // Vᵀ: each vector a row
      if constexpr (std::is_floating_point_v<T>) {
        buffer<double> re(n_, 0.0);
        buffer<double> im(n_, 0.0);
        for (std::size_t i {0}; i < n_; ++i) {
          re[i] = a[(i * n_) + i]; // the isolated ones; the active block's are overwritten
        }
        detail::schur_form(target, re, im, steps);
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
        detail::schur_form(target, values, steps);
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
      // Vᵀ ← Vᵀ D, then the exchanges undone, on its columns, each row normalised; then a pair's second row, the
      // first's conjugate. Each row its own, on the threads.
      detail::parallel_for(n_, detail::memory_element_weight * 4 * n_, [&](std::size_t c) {
                             std::complex<double>* row {vt.data() + (c * n_)};
                             for (std::size_t i {balanced.low}; i < balanced.high; ++i) {
                               row[i] *= balanced.scale[i];
                             }
                             for (std::size_t s {balanced.swaps.size()}; s-- > 0;) {
                               std::swap(row[balanced.swaps[s].first], row[balanced.swaps[s].second]);
                             }
                             finish(row, values_[c], T {});
                           });
      detail::parallel_for(n_, detail::memory_element_weight * n_, [&](std::size_t c) { conjugate_second(vt.data() + (c * n_), values_[c], T {}); });
      vectors_ = transpose(matrix<std::complex<double>> {n_, n_, std::move(vt)});
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
    //!        real one to unit norm; a pair's first to unit norm, its largest component real (its second
    //!        \ref conjugate_second's).
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
      else if (value.imag() > 0.0) {
        detail::normalize_vector(row, n_, true);
      }
    }

    //! \brief A real matrix's pair's second eigenvector \p row (\p value's imaginary part negative): the first's, the
    //!        row above, conjugated, once \ref finish has normalised it; nothing for any other.
    void conjugate_second(std::complex<double>*       row,
                          const std::complex<double>& value,
                          double /*real*/) const
    {
      if (value.imag() < 0.0) {
        const std::complex<double>* first {row - n_};
        for (std::size_t r {0}; r < n_; ++r) {
          row[r] = std::conj(first[r]);
        }
      }
    }

    //! \brief No pairs in a complex matrix's eigenvectors.
    void conjugate_second(std::complex<double>* /*row*/,
                          const std::complex<double>& /*value*/,
                          const std::complex<double>& /*complex*/) const noexcept
    {
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
