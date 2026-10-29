/*!
 * \file svd.hpp
 * \brief The singular value decomposition and what it answers: the singular values, least squares, the
 *        pseudo-inverse and the numerical rank. The bidiagonal reduction and its iterations, divide and conquer and
 *        one-sided Jacobi live here, apart from the factorisations they build on (factorizations.hpp).
 */
#ifndef NUMERICAL_SVD_HPP
#define NUMERICAL_SVD_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "concepts.hpp"
#include "dqds.hpp"
#include "factorizations.hpp"

namespace numerical {

  namespace detail {

    //! \brief The sum of squares under which one-sided Jacobi judges a pair of columns over their norms: below it a
    //!        square has lost digits to underflow (`DBL_MIN / ε`).
    inline constexpr double jacobi_tiny {std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon()};

    //! \brief The |ζ| past which one-sided Jacobi takes the rotation's tangent as `1 / (2|ζ|)`: ζ² stays finite under it.
    inline constexpr double jacobi_large {1e150};

    /*!
     * \brief One-sided (Hestenes) Jacobi on the columns of the `n × n` row-major \p w: plane rotations,
     *        accumulated into \p v, until every pair of columns is orthogonal to rounding. The column
     *        norms are then the singular values of the input, each to high relative accuracy.
     * \param[in,out] w          The matrix whose columns are orthogonalised.
     * \param[in,out] v          The rotations, accumulated (start from the identity).
     * \param[in]     n          The order.
     * \param[in]     max_sweeps The bound of sweeps.
     * \throws convergence_error If a sweep still rotated at the bound.
     */
    template <typename T>
    void one_sided_jacobi(buffer<T>&      w,
                          buffer<T>&      v,
                          std::size_t     n,
                          std::size_t     max_sweeps)
    {
      // The sweeps rotate columns: work on the transposes, where each column is a contiguous row. A walk
      // down the columns of the row-major arrays strides by n on every access and runs several times slower.
      const auto transpose_square {[n](buffer<T>& m) {
                                     for (std::size_t i {0}; i < n; ++i) {
                                       for (std::size_t j {i + 1}; j < n; ++j) {
                                         std::swap(m[(i * n) + j], m[(j * n) + i]);
                                       }
                                     }
                                   }};
      transpose_square(w);
      transpose_square(v);
      // Orthogonal to the rounding of the columns' storage: a float column holds no inner product to double's.
      // The rotations are carried in double: the coarser of T's epsilon and double's.
      const double epsilon {std::max(static_cast<double>(std::numeric_limits<value_range_t<T>>::epsilon()), std::numeric_limits<double>::epsilon())};
      for (std::size_t sweep {1};; ++sweep) {
        bool rotated {false};
        for (std::size_t p {0}; p < n; ++p) {
          for (std::size_t q {p + 1}; q < n; ++q) {
            const T* wp    {w.data() + (p * n)};
            const T* wq    {w.data() + (q * n)};
            double   alpha {0.0};
            double   beta  {0.0};
            T        gamma {};
            for (std::size_t i {0}; i < n; ++i) {
              alpha += static_cast<double>(std::norm(wp[i]));
              beta  += static_cast<double>(std::norm(wq[i]));
              gamma += conj_of(wp[i]) * wq[i];
            }
            double magnitude {static_cast<double>(std::abs(gamma))};
            double zeta      {};
            if (std::min(alpha, beta) < jacobi_tiny) {
              // A column whose squares underflow (rows of a far smaller scale than the largest): its sum of squares
              // is 0 or has lost its digits, and the test below never holds. The pair is judged on the columns over
              // their norms, which need no squares: the cosine, and ζ from the ratio of the norms.
              const auto norm_p {static_cast<double>(scaled_norm(wp, n))};
              const auto norm_q {static_cast<double>(scaled_norm(wq, n))};
              if (norm_p == 0.0 || norm_q == 0.0) {
                continue; // a zero column is orthogonal to any other
              }
              T cosine {};
              for (std::size_t i {0}; i < n; ++i) {
                cosine += conj_of(wp[i] / detail::from_carried<T>(norm_p)) * (wq[i] / detail::from_carried<T>(norm_q));
              }
              const double size {static_cast<double>(std::abs(cosine))};
              zeta = ((norm_q / norm_p) - (norm_p / norm_q)) / (2.0 * size);
              if (size <= epsilon) {
                continue; // orthogonal to rounding
              }
              gamma     = cosine; // the phase is γ's
              magnitude = size;
            }
            else {
              if (magnitude <= epsilon * std::sqrt(alpha) * std::sqrt(beta)) {
                continue; // already orthogonal to rounding (or a zero column)
              }
              zeta = (beta - alpha) / (2.0 * magnitude);
            }
            // Past jacobi_large, ζ² overflows and t would be 0 (no rotation, ever): there |ζ| + √(1 + ζ²) is 2|ζ| exactly,
            // and t is 1 / (2|ζ|). A tangent that still underflows to 0 is a rotation below the smallest double.
            const double t {(zeta >= 0.0 ? 1.0 : -1.0) * (std::abs(zeta) > jacobi_large ? 0.5 / std::abs(zeta) : 1.0 / (std::abs(zeta) + std::sqrt(1.0 + (zeta * zeta))))};
            if (t == 0.0) {
              continue; // a rotation below the smallest double: the pair is as orthogonal as it can be
            }
            rotated = true;
            // Turn column q by the phase of γ so that the pair's inner product is real, then rotate.
            const T      turn {conj_of(gamma) / detail::from_carried<T>(magnitude)};
            const double c    {1.0 / std::sqrt(1.0 + (t * t))};
            const double s    {c * t};
            for (buffer<T>* side : {&w, &v}) {
              T* mp {side->data() + (p * n)};
              T* mq {side->data() + (q * n)};
              for (std::size_t i {0}; i < n; ++i) {
                const T left  {mp[i]};
                const T right {turn * mq[i]};
                mp[i] = (detail::from_carried<T>(c) * left) - (detail::from_carried<T>(s) * right);
                mq[i] = (detail::from_carried<T>(s) * left) + (detail::from_carried<T>(c) * right);
              }
            }
          }
        }
        if (!rotated) {
          transpose_square(w);
          transpose_square(v);
          return;
        }
        if (sweep == max_sweeps) {
          throw convergence_error("SVD: one-sided Jacobi did not converge within its sweeps");
        }
      }
    }

    //! \brief The QR steps the bidiagonal singular value iteration may take per value: it converges in two
    //!        or three; the bound turns a non-convergence into an error instead of a hang.
    inline constexpr std::size_t svd_qr_steps_per_value {30};

    //! \brief The order below which a real bidiagonal reduction goes a reflection at a time, as LAPACK's `dgebrd`
    //!        falls back to `dgebd2`: the panels' products are too small to pay for packing. Order 64: 33 % shorter
    //!        on six x86-64 cores, 7 % on arm64; order 160: 30 % longer on arm64, where x86-64 still gains 8 %.
    inline constexpr std::size_t bidiagonal_crossover {128};

    //! \brief The order below which the bases of a bidiagonal reduction are formed a reflector at a time: the blocks'
    //!        packing and triangular factor cost more than they save. Order 32: 11 % shorter on six x86-64 cores,
    //!        12 % on arm64; order 96: 7 % longer on both.
    inline constexpr std::size_t bidiagonal_basis_crossover {64};

    /*!
     * \brief The bases of a bidiagonal reduction from its reflectors: `U_b = (H₀ H₁ …) diag(conj left phases)` and
     *        `V_b = (H'₀ H'₁ …) diag(right phases)`, each formed by blocks of reflectors through the matrix product
     *        kernel, as QR's `Q` is, or a reflector at a time below \ref bidiagonal_basis_crossover.
     * \param[in]  left        The left reflector of each step (acting from its row), empty where none.
     * \param[in]  right       The right reflector of each step (acting from the column past it), likewise.
     * \param[in]  left_tau    Each left reflector's `τ` (`I − τ w wᴴ`, \ref larfg_in_place).
     * \param[in]  right_tau   Each right reflector's.
     * \param[in]  left_phase  The unit phase each row took.
     * \param[in]  right_phase The unit phase each column past the diagonal took.
     * \param[in]  n           The order.
     * \param[out] ut          `U_bᵀ`, row-major: its rows are the columns of `U_b`.
     * \param[out] vt          `V_bᵀ`, likewise.
     */
    template <typename T>
    void form_bidiagonal_bases(const std::vector<buffer<T>>&      left,
                               std::vector<buffer<T>>             right,
                               const std::vector<T>&              left_tau,
                               const std::vector<T>&              right_tau,
                               const buffer<T>&                   left_phase,
                               const buffer<T>&                   right_phase,
                               std::size_t                        n,
                               buffer<T>&                         ut,
                               buffer<T>&                         vt)
    {
      // U_b = L₀ᴴ L₁ᴴ … with L_kᴴ = H_k diag(conj phase at k), and V_b = R₀ R₁ … with R_k = H'_k diag(phase
      // at k+1). A phase on row k commutes with every later reflector, which acts on rows past it, so
      // U_b = (H₀ H₁ …) diag(conj phases) and V_b = (H'₀ H'₁ …) diag(phases): the reflectors are applied to the
      // diagonal by blocks, through the matrix product kernel, as QR's Q is formed.
      ut.assign(n * n, T {}); // Uᵀ and Vᵀ, formed transposed: diagonal to start, the reflectors applied to them
      vt.assign(n * n, T {});
      for (std::size_t k {0}; k < n; ++k) {
        ut[(k * n) + k] = conj_of(left_phase[k]);
        vt[(k * n) + k] = k == 0 ? T {1} : right_phase[k - 1];
      }
      std::vector<buffer<T>> shifted(n); // right[k] acts from row k+1: as reflector k+1 of a QR
      std::vector<T>         shifted_tau(n, T {});
      for (std::size_t k {0}; k + 1 < n; ++k) {
        shifted[k + 1]     = std::move(right[k]);
        shifted_tau[k + 1] = right_tau[k];
      }
      if (n < bidiagonal_basis_crossover) { // a reflector at a time, the last first, as LAPACK's dorg2r
        for (std::size_t k {n}; k-- > 0;) {
          reflect_columns(ut.data(), n, left[k], k, k, n, left_tau[k]);
          reflect_columns(vt.data(), n, shifted[k], k, k, n, shifted_tau[k]);
        }
        return;
      }
      for (std::size_t k1 {n}; k1 > 0;) { // blocks of qr_block reflectors, the last first
        const std::size_t k0 {(k1 - 1) / qr_block * qr_block};
        apply_transposed_reflector_block(gather_reflectors(left, k0, k1, n, &left_tau), ut, n, k0, n);
        apply_transposed_reflector_block(gather_reflectors(shifted, k0, k1, n, &shifted_tau), vt, n, k0, n);
        k1 = k0;
      }
    }

    //! \brief The rows from which a step of the blocked bidiagonal reduction shares its coefficients and corrections
    //!        with the threads: on x86-64 (six AVX2 cores, the reduction of order 512 −5 to −14 %, 1024 −12 %); never
    //!        on arm64, where its serial part is smaller and the shared step gained nothing (M1 Pro, eight cores,
    //!        order 1024 0.97 to 1.05 times as long).
#ifdef NUMERICAL_X86_COMPILED
    inline constexpr std::size_t bidiagonal_shared_least {256};
#endif

    //! \brief The reflections from each side in one panel of the blocked bidiagonal reduction.
    inline constexpr std::size_t bidiagonal_block {32};

    //! \brief The weight of a multiply-add of the bidiagonal reduction's matrix-vector products, which read the trailing
    //!        block once for each column: on x86-64 a thread shares them from a quarter of the work (six cores,
    //!        singular values of order 256: 8.1 → 7.3 ms, 512: 35 → 30); on arm64 that slowed order 256 by 10 %.
#ifdef NUMERICAL_X86_COMPILED
    inline constexpr std::size_t bidiagonal_product_weight {4};
#else
    inline constexpr std::size_t bidiagonal_product_weight {1};
#endif

    template <typename T>
    void bidiagonalize_unblocked(buffer<T>&           b,
                                 std::size_t          n,
                                 buffer<double>&      d,
                                 buffer<double>&      e,
                                 buffer<T>&           ut,
                                 buffer<T>&           vt,
                                 bool                 form = true);

    /*!
     * \brief Reduces the square `n × n` row-major \p b to upper-bidiagonal form, `B = U_b D V_bᴴ`, by panels
     *        of \ref bidiagonal_block reflections from each side, as LAPACK's `dgebrd` and `dlabrd` (`zgebrd`,
     *        `zlabrd`): the matrix a panel's later steps see is `B − V Ȳᵀ − X Ūᵀ` (`V`, `U` the reflectors, `Y`, `X`
     *        what they move, `Ȳ` and `Ū` held conjugated, so a complex matrix's update is a plain product), applied
     *        to the one column and the one row each step needs; the trailing block takes it after the panel, through
     *        the matrix product kernel. A real bidiagonal keeps its signs, which the iterations that follow take; a
     *        complex one is turned real as \ref bidiagonalize_unblocked turns it, a unit phase on row `k` before
     *        its right reflection and on column `k + 1` once that column comes to be reflected.
     * \param[in,out] b  The matrix; destroyed.
     * \param[in]     n  Its order.
     * \param[out]    d  The diagonal (size \p n).
     * \param[out]    e  The superdiagonal in `e[0…n-2]`, and `e[n-1] = 0` (size \p n).
     * \param[out]    ut `U_bᵀ`, row-major: its rows are the columns of `U_b` (left as it is without \p form).
     * \param[out]    vt `V_bᵀ`, row-major: its rows are the columns of `V_b` (likewise).
     * \param[in]     form Whether to form \p ut and \p vt; the bidiagonal alone gives the singular values.
     */
    template <typename T>
    requires std::is_floating_point_v<T> || split_complex<T>
    void bidiagonalize(buffer<T>&           b,
                       std::size_t          n,
                       buffer<double>&      d,
                       buffer<double>&      e,
                       buffer<T>&           ut,
                       buffer<T>&           vt,
                       bool                 form = true)
    {
      if (n < bidiagonal_crossover) {
        bidiagonalize_unblocked(b, n, d, e, ut, vt, form);
        return;
      }
      std::vector<buffer<T>>      left(n);               // the left reflector of step k, on rows k…, empty where none was needed
      std::vector<buffer<T>>      right(n);              // the right reflector of step k, on columns k+1…, likewise
      std::vector<T>              left_tau(n, T {});     // each one's τ, as LAPACK's dlarfg forms them (larfg_in_place)
      std::vector<T>              right_tau(n, T {});
      buffer<T>                   left_phase(n, T {1});  // row k times it puts |d_k| on the diagonal (1: a real T)
      buffer<T>                   right_phase(n, T {1}); // column k + 1 times it puts |e_k| above
      const auto                  nonzero {[](const T& value) { return value != T {}; }};
      constexpr bool              complex_entries {split_complex<T>};
      // By reflector, each row over the whole order: row 2p of vx holds step k0 + p's left reflector v (at its rows),
      // row 2p + 1 what its right one moves, x = 2 B u (at the rows); row 2p of yu what v moves, y = 2 Bᵀ v (at the
      // columns), row 2p + 1 the right reflector u (at its columns). A step's terms of the panel's earlier ones are then
      // one update through their rows (subtract_products, a family's rows 2n apart), in their order.
      buffer<T> vx(2 * bidiagonal_block * n);
      buffer<T> yu(2 * bidiagonal_block * n);
      buffer<T> columns(bidiagonal_block * n);      // the panel's columns of B, gathered row by row
      buffer<T> factors(2 * bidiagonal_block);      // a step's coefficients of those rows
#ifdef NUMERICAL_X86_COMPILED
      buffer<T> next_factors(2 * bidiagonal_block); // the next column's, applied ahead (a shared step)
#endif
      for (std::size_t k0 {0}; k0 < n; k0 += bidiagonal_block) {
        const std::size_t k1    {std::min(k0 + bidiagonal_block, n)};
        const std::size_t width {k1 - k0};
        std::fill_n(vx.data(), 2 * width * n, T {});
        std::fill_n(yu.data(), 2 * width * n, T {});
        for (std::size_t r {k0}; r < n; ++r) { // column k from row k down, each row's part read once
          for (std::size_t k {k0}; k < std::min(k1, r + 1); ++k) {
            columns[((k - k0) * n) + r] = b[(r * n) + k];
          }
        }
        const auto v_row {[&vx, n](std::size_t p, std::size_t from) { return vx.data() + (2 * p * n) + from; }};
        const auto x_row {[&vx, n](std::size_t p, std::size_t from) { return vx.data() + (((2 * p) + 1) * n) + from; }};
        const auto y_row {[&yu, n](std::size_t p, std::size_t from) { return yu.data() + (2 * p * n) + from; }};
        const auto u_row {[&yu, n](std::size_t p, std::size_t from) { return yu.data() + (((2 * p) + 1) * n) + from; }};
#ifdef NUMERICAL_X86_COMPILED
        bool column_ahead {false}; // whether the step's column took the steps before the last ahead (a shared step)
#endif
        for (std::size_t k {k0}; k < k1; ++k) {
          const std::size_t i    {k - k0};
#ifdef NUMERICAL_X86_COMPILED
          const std::size_t from {column_ahead ? i - 1 : 0};
          column_ahead = false;
#else
          constexpr std::size_t from {0};
#endif
          for (std::size_t p {from}; p < i; ++p) { // v's row by y's element, x's by u's
            factors[2 * (p - from)]       = *y_row(p, k);
            factors[(2 * (p - from)) + 1] = *u_row(p, k);
          }
          T* const column {columns.data() + (i * n) + k}; // column k from the diagonal down, the panel's earlier steps applied
          subtract_products(column, factors.data(), v_row(from, k), n, 2 * (i - from), n - k);
          if constexpr (complex_entries) {
            if (k > 0) { // the previous step's phase on this column, now that it is to be reflected
              for (std::size_t r {0}; r < n - k; ++r) {
                column[r] *= right_phase[k - 1];
              }
            }
          }
          buffer<T> x(column, column + (n - k));
          if (std::any_of(x.begin() + 1, x.end(), nonzero)) { // something below the diagonal to reflect
            const auto [beta, tau] {larfg_in_place(x)};       // Hᴴ = I − τ̄ w wᴴ from the left, as LAPACK's dlabrd
            if constexpr (complex_entries) {
              d[k]          = static_cast<double>(std::abs(beta));
              left_phase[k] = conj_of(phase_of(beta));
            }
            else {
              d[k] = static_cast<double>(beta);
            }
            // y = τ (B − V Yᵀ − X Uᵀ)ᴴ v on the columns past k, held conjugated: Bᵀ v̄ by contiguous rows, then the
            // panel's terms, times τ̄.
            buffer<T> by_vx(2 * i, T {});  // Vᵀ v and Xᵀ v over the earlier steps, interleaved as their rows
            buffer<T> conjugate(x.size()); // v̄: Bᴴ v as the terms of (Bᵀ v̄) (v itself for a real T)
            std::transform(x.begin(), x.end(), conjugate.begin(), [](const T& value) { return conj_of(value); });
            for (std::size_t p {0}; p < i; ++p) {
              by_vx[2 * p]       = dot_product(v_row(p, k), conjugate.data(), n - k);
              by_vx[(2 * p) + 1] = dot_product(x_row(p, k), conjugate.data(), n - k);
            }
            const std::size_t     span        {n - k - 1};
            constexpr std::size_t chunk_width {64};
            // Bᵀ v as the terms −(−v)·B, the chunk held through them (negation is exact, so each column takes the same
            // additions): row by row, each term stored the chunk back (singular values of order 512 on one x86-64
            // core: 46 → 42 ms).
            buffer<T> minus(x.size(), T {});
            std::transform(conjugate.begin(), conjugate.end(), minus.begin(), std::negate<T> {});
            parallel_for((span + chunk_width - 1) / chunk_width, bidiagonal_product_weight * (n - k) * chunk_width, [&](std::size_t chunk) {
                           const std::size_t first {k + 1 + (chunk * chunk_width)};
                           const std::size_t count {std::min(chunk_width, n - first)};
                           buffer<T>    sum(count, T {});
                           subtract_products(sum.data(), minus.data(), b.data() + (k * n) + first, n, n - k, count);
                           subtract_products(sum.data(), by_vx.data(), y_row(0, first), n, 2 * i, count);
                           for (std::size_t c {0}; c < count; ++c) {
                             *y_row(i, first + c) = conj_of(tau) * sum[c];
                           }
                         });
            std::copy(x.begin(), x.end(), v_row(i, k));
            left[k]     = std::move(x);
            left_tau[k] = tau;
          }
          else if constexpr (complex_entries) {
            d[k]          = static_cast<double>(std::abs(x[0]));
            left_phase[k] = conj_of(phase_of(x[0]));
          }
          else {
            d[k] = static_cast<double>(x[0]);
          }
          if (k + 1 == n) {
            break;
          }
          buffer<T> y(b.begin() + static_cast<std::ptrdiff_t>((k * n) + k + 1), // row k past the diagonal, the left
                      b.begin() + static_cast<std::ptrdiff_t>((k + 1) * n));    // reflection included
          for (std::size_t p {0}; p <= i; ++p) {
            factors[p] = *v_row(p, k);
          }
          subtract_products(y.data(), factors.data(), y_row(0, k + 1), 2 * n, i + 1, n - k - 1);
          for (std::size_t p {0}; p < i; ++p) {
            factors[p] = *x_row(p, k);
          }
          subtract_products(y.data(), factors.data(), u_row(0, k + 1), 2 * n, i, n - k - 1);
          if constexpr (complex_entries) { // the row times its phase, then conjugated: B ← B H with H yᴴ ∥ e₁
            for (T& element : y) {
              element = conj_of(element * left_phase[k]);
            }
          }
          if (std::any_of(y.begin() + 1, y.end(), nonzero)) { // something past the superdiagonal to reflect
            const auto [beta, tau] {larfg_in_place(y)};       // B ← B (I − τ u uᴴ), as LAPACK's dlabrd
            if constexpr (complex_entries) { // the row's entry is β̄: its phase's conjugate turns it to |β|
              e[k]           = static_cast<double>(std::abs(beta));
              right_phase[k] = phase_of(beta);
            }
            else {
              e[k] = static_cast<double>(beta);
            }
            // x = τ (B − V Yᵀ − X Uᵀ) u on the rows past k: B u by rows, then the panel's terms.
            const std::size_t rows          {n - k - 1};
            buffer<T>         by_y(i + 1, T {}); // Yᵀ u over this and the earlier steps, Uᵀ u over the earlier
            buffer<T>         by_u(i, T     {});
            buffer<T>         moved(rows, T {});
#ifdef NUMERICAL_X86_COMPILED
            if (rows >= bidiagonal_shared_least) {
              // The coefficients run among B u's tasks, and the corrections by bands on the threads, the next column
              // taking the earlier steps' terms in the same pass (its rows are these): each element the operations,
              // in the order, of the branch above.
              parallel_for(rows + (2 * i) + 1, bidiagonal_product_weight * 2 * (n - k), [&](std::size_t task) {
                             if (task < rows) {
                               moved[task] = dot_product(b.data() + ((k + 1 + task) * n) + k + 1, y.data(), y.size());
                             }
                             else if (task - rows <= i) {
                               by_y[task - rows] = dot_product(y_row(task - rows, k + 1), y.data(), y.size());
                             }
                             else {
                               by_u[task - rows - i - 1] = dot_product(u_row(task - rows - i - 1, k + 1), y.data(), y.size());
                             }
                           });
              T* const next {k + 1 < k1 && i > 0 ? columns.data() + ((i + 1) * n) + k + 1 : nullptr};
              for (std::size_t p {0}; next != nullptr && p < i; ++p) { // the next column's terms of the steps before this
                next_factors[2 * p]       = *y_row(p, k + 1);
                next_factors[(2 * p) + 1] = *u_row(p, k + 1);
              }
              constexpr std::size_t band {256};
              parallel_for((rows + band - 1) / band, band * 4 * (i + 1) * 8, [&](std::size_t c) {
                             const std::size_t first {c * band};
                             const std::size_t count {std::min(band, rows - first)};
                             subtract_products(moved.data() + first, by_y.data(), v_row(0, k + 1 + first), 2 * n, i + 1, count);
                             subtract_products(moved.data() + first, by_u.data(), x_row(0, k + 1 + first), 2 * n, i, count);
                             for (std::size_t r {first}; r < first + count; ++r) {
                               *x_row(i, k + 1 + r) = tau * moved[r];
                             }
                             if (next != nullptr) {
                               subtract_products(next + first, next_factors.data(), v_row(0, k + 1 + first), n, 2 * i, count);
                             }
                           });
              column_ahead = next != nullptr;
            }
            else
#endif
            {
              for (std::size_t p {0}; p <= i; ++p) {
                by_y[p] = dot_product(y_row(p, k + 1), y.data(), y.size());
              }
              for (std::size_t p {0}; p < i; ++p) {
                by_u[p] = dot_product(u_row(p, k + 1), y.data(), y.size());
              }
              parallel_for(rows, bidiagonal_product_weight * 2 * (n - k), [&](std::size_t row) {
                             moved[row] = dot_product(b.data() + ((k + 1 + row) * n) + k + 1, y.data(), y.size());
                           });
              subtract_products(moved.data(), by_y.data(), v_row(0, k + 1), 2 * n, i + 1, rows);
              subtract_products(moved.data(), by_u.data(), x_row(0, k + 1), 2 * n, i, rows);
              for (std::size_t r {0}; r < rows; ++r) {
                *x_row(i, k + 1 + r) = tau * moved[r];
              }
            }

            std::transform(y.begin(), y.end(), u_row(i, k + 1), [](const T& value) { return conj_of(value); }); // Ū
            right[k]     = std::move(y);
            right_tau[k] = tau;
          }
          else if constexpr (complex_entries) {                                                                 // y₀ the entry's conjugate
            e[k]           = static_cast<double>(std::abs(y[0]));
            right_phase[k] = phase_of(y[0]);
          }
          else {
            e[k] = static_cast<double>(y[0]);
          }
        }
        if (k1 < n) { // the trailing block from row and column k1: B −= V Yᵀ + X Uᵀ
          const std::size_t rest                    {n - k1};
          buffer<T>         minus_v(rest * width, T {});
          buffer<T>         minus_x(rest * width, T {});
          for (std::size_t r {0}; r < rest; ++r) {
            for (std::size_t p {0}; p < width; ++p) {
              minus_v[(r * width) + p] = -*v_row(p, k1 + r);
              minus_x[(r * width) + p] = -*x_row(p, k1 + r);
            }
          }
          T* trailing {b.data() + (k1 * n) + k1};
          gemm(rest, rest, width, minus_v.data(), width, y_row(0, k1), 2 * n, trailing, n);
          gemm(rest, rest, width, minus_x.data(), width, u_row(0, k1), 2 * n, trailing, n);
        }
      }
      e[n - 1] = 0.0;
      if (form) {
        form_bidiagonal_bases(left, std::move(right), left_tau, right_tau, left_phase, right_phase, n, ut, vt);
      }
    }

    /*!
     * \brief Reduces the square `n × n` row-major \p b to real upper-bidiagonal form, `B = U_b D V_bᴴ` with
     *        `D` real, by Householder reflections alternately from the left and the right (Golub–Kahan).
     *
     * A complex entry left on the bidiagonal is turned real by a unit phase on its row or column, as LAPACK
     * does, so the iteration that follows needs only real rotations.
     * \param[in,out] b  The matrix; destroyed.
     * \param[in]     n  Its order.
     * \param[out]    d  The diagonal (size \p n).
     * \param[out]    e  The superdiagonal in `e[0…n-2]`, and `e[n-1] = 0` (size \p n).
     * \param[out]    ut `U_bᵀ`, row-major: its rows are the columns of `U_b` (left as it is without \p form).
     * \param[out]    vt `V_bᵀ`, row-major: its rows are the columns of `V_b` (likewise).
     * \param[in]     form Whether to form \p ut and \p vt.
     */
    template <typename T>
    void bidiagonalize_unblocked(buffer<T>&           b,
                                 std::size_t          n,
                                 buffer<double>&      d,
                                 buffer<double>&      e,
                                 buffer<T>&           ut,
                                 buffer<T>&           vt,
                                 bool                 form)
    {
      std::vector<buffer<T>>      left(n);  // the left reflector of step k, empty where none was needed
      std::vector<buffer<T>>      right(n); // the right reflector acting on columns k+1…, likewise
      std::vector<T>              left_tau(n, T    {});
      std::vector<T>              right_tau(n, T   {});
      buffer<T>                   left_phase(n, T  {1});
      buffer<T>                   right_phase(n, T {1});
      for (std::size_t k {0}; k < n; ++k) {
        buffer<T> x(n - k, T {}); // column k from the diagonal down
        for (std::size_t i {k}; i < n; ++i) {
          x[i - k] = b[(i * n) + k];
        }
        if (scaled_norm(x.data() + 1, x.size() - 1) > 0.0) { // as LAPACK's dgebd2: Hᴴ = I − τ̄ w wᴴ from the left
          const auto [beta, tau] {larfg_in_place(x)};
          reflect_rows(b, n, x, k, k + 1, conj_of(tau));
          b[(k * n) + k] = beta;
          left[k]        = std::move(x);
          left_tau[k]    = tau;
        }
        left_phase[k] = conj_of(phase_of(b[(k * n) + k])); // row k times it puts |b_kk| on the diagonal
        for (std::size_t j {k}; j < n; ++j) {
          b[(k * n) + j] *= left_phase[k];
        }
        d[k] = static_cast<double>(std::abs(b[(k * n) + k]));
        if (k + 1 == n) {
          break;
        }
        buffer<T> y(n - k - 1, T {}); // row k right of the diagonal, conjugated: B ← B H with H yᴴ ∥ e₁
        for (std::size_t j {k + 1}; j < n; ++j) {
          y[j - k - 1] = conj_of(b[(k * n) + j]);
        }
        if (y.size() > 1 && scaled_norm(y.data() + 1, y.size() - 1) > 0.0) {
          const auto [beta, tau] {larfg_in_place(y)};
          parallel_for(n - k - 1, 2 * y.size(), [&](std::size_t r) { // row ← row (I − τ y yᴴ); rows above k are zero there
                         T*      row   {b.data() + ((k + 1 + r) * n) + k + 1};
                         const T scale {tau * dot_product(row, y.data(), y.size())};
                         for (std::size_t j {0}; j < y.size(); ++j) {
                           row[j] -= scale * conj_of(y[j]);
                         }
                       });
          b[(k * n) + k + 1] = beta; // row k is β e₁ past the diagonal: the rest is read no more
          right[k]           = std::move(y);
          right_tau[k]       = tau;
        }
        right_phase[k] = conj_of(phase_of(b[(k * n) + k + 1])); // column k+1 times it: |b_k,k+1| above
        for (std::size_t i {k}; i < n; ++i) {
          b[(i * n) + k + 1] *= right_phase[k];
        }
        e[k] = static_cast<double>(std::abs(b[(k * n) + k + 1]));
      }
      if (n > 0) {
        e[n - 1] = 0.0;
      }
      if (form) {
        form_bidiagonal_bases(left, std::move(right), left_tau, right_tau, left_phase, right_phase, n, ut, vt);
      }
    }

    //! \brief \ref bidiagonalize_unblocked, for any `T` the panels do not take (a complex `long double`).
    template <typename T>
    requires(!std::is_floating_point_v<T> && !split_complex<T>)
    void bidiagonalize(buffer<T>&           b,
                       std::size_t          n,
                       buffer<double>&      d,
                       buffer<double>&      e,
                       buffer<T>&           ut,
                       buffer<T>&           vt,
                       bool                 form = true)
    {
      bidiagonalize_unblocked(b, n, d, e, ut, vt, form);
    }

    /*!
     * \brief The singular values of the real upper bidiagonal (\p d, \p e) by implicit-shift QR with
     *        deflation (the Golub–Kahan–Reinsch iteration, LINPACK's `dsvdc`), the rotations accumulated
     *        into the rows of \p ut and \p vt. Each block's bulge is chased from its larger end, as LAPACK's `dbdsqr`
     *        chases it: from the top only, a block graded up (its small entries first) never converged.
     * \param[in,out] d         The diagonal; on return, the singular values, descending and non-negative.
     * \param[in,out] e         The superdiagonal in `e[0…n-2]`, `e[n-1] = 0`; destroyed.
     * \param[in,out] ut        Row-major `n × n`; on return, row `j` is the left vector of `d[j]`. Empty, with \p vt:
     *                          the singular values alone.
     * \param[in,out] vt        Row-major `n × n`; on return, row `j` is the right vector of `d[j]`.
     * \param[in]     n         The order.
     * \param[in]     max_steps The bound of QR steps in all.
     * \throws convergence_error If the iteration has not converged after \p max_steps steps.
     */
    template <typename T>
    void bidiagonal_qr(buffer<double>&      d,
                       buffer<double>&      e,
                       buffer<T>&           ut,
                       buffer<T>&           vt,
                       std::size_t          n,
                       std::size_t          max_steps)
    {
      // The iteration reads only d and e: what it does to the rows of vt and ut is recorded and applied once
      // at the end, in parallel by columns.
      std::vector<row_operation> on_vt;
      std::vector<row_operation> on_ut;
      const bool                 record      {!vt.empty()}; // the singular values alone record nothing
      const double               epsilon     {std::numeric_limits<double>::epsilon()};
      const double               tiny        {std::numeric_limits<double>::min() / epsilon};
      constexpr std::size_t      none        {std::numeric_limits<std::size_t>::max()};
      std::size_t                p           {n};
      std::size_t                steps       {0};
      bool                       down        {true}; // the direction of the chase, chosen per block
      std::size_t                chased_top  {n};
      std::size_t                chased_last {0};
      // One implicit QR step on the block top…p−1 of the bidiagonal D, E (the entries of d and e as \p at maps their
      // positions), shifted by the eigenvalue of the trailing 2 × 2 of BᵀB nearer its last diagonal entry: the bulge
      // chased from top to bottom, its rotations recorded on the right and on the left.
      const auto qr_step {[&](std::size_t first, std::size_t end, auto&& dd, auto&& ee, auto&& at, std::vector<row_operation>& right,
                              std::vector<row_operation>& left) {
                            const double scale {std::max({std::abs(dd(end - 1)), std::abs(dd(end - 2)), std::abs(ee(end - 2)),
                                                          std::abs(dd(first)), std::abs(ee(first))})};
                            const double sp    {dd(end - 1) / scale};
                            const double spm1  {dd(end - 2) / scale};
                            const double epm1  {ee(end - 2) / scale};
                            const double sk    {dd(first) / scale};
                            const double ek    {ee(first) / scale};
                            const double bb    {(((spm1 + sp) * (spm1 - sp)) + (epm1 * epm1)) / 2.0};
                            const double cc    {(sp * epm1) * (sp * epm1)};
                            double       shift {0.0};
                            if (bb != 0.0 || cc != 0.0) {
                              shift = std::sqrt((bb * bb) + cc);
                              shift = cc / (bb < 0.0 ? bb - shift : bb + shift);
                            }
                            double f {((sk + sp) * (sk - sp)) + shift};
                            double g {sk * ek};
                            for (std::size_t j {first}; j + 1 < end; ++j) {
                              double t  {rotation_length(f, g)};
                              double cs {f / t};
                              double sn {g / t};
                              if (j != first) {
                                ee(j - 1) = t;
                              }
                              f         = (cs * dd(j)) + (sn * ee(j));
                              ee(j)     = (cs * ee(j)) - (sn * dd(j));
                              g         = sn * dd(j + 1);
                              dd(j + 1) = cs * dd(j + 1);
                              if (record) {
                                right.push_back({row_operation::kind::rotate, at(j), at(j + 1), cs, sn});
                              }
                              t         = rotation_length(f, g);
                              cs        = f / t;
                              sn        = g / t;
                              dd(j)     = t;
                              f         = (cs * ee(j)) + (sn * dd(j + 1));
                              dd(j + 1) = (cs * dd(j + 1)) - (sn * ee(j));
                              if (j + 2 < end) { // past the block, e is zero and stays so
                                g         = sn * ee(j + 1);
                                ee(j + 1) = cs * ee(j + 1);
                              }
                              if (record) {
                                left.push_back({row_operation::kind::rotate, at(j), at(j + 1), cs, sn});
                              }
                            }
                            ee(end - 2) = f;
                          }};
      while (p > 0) {
        // The largest k < p−1 with a negligible e[k] (none: the loop ran out and wrapped).
        std::size_t k {p - 1};
        while (k-- > 0) {
          if (std::abs(e[k]) <= tiny + (epsilon * (std::abs(d[k]) + std::abs(d[k + 1])))) {
            e[k] = 0.0;
            break;
          }
        }
        const std::size_t top {k == none ? 0 : k + 1}; // the unreduced block is top…p−1
        if (top == p - 1) { // d[p−1] has converged: make it non-negative, sort it into place
          if (d[top] < 0.0) {
            d[top] = -d[top];
            if (record) {
              on_vt.push_back({row_operation::kind::negate, top, top, 0.0, 0.0});
            }
          }
          for (std::size_t j {top}; j + 1 < n && d[j] < d[j + 1]; ++j) {
            std::swap(d[j], d[j + 1]);
            if (record) {
              on_vt.push_back({row_operation::kind::swap, j, j + 1, 0.0, 0.0});
              on_ut.push_back({row_operation::kind::swap, j, j + 1, 0.0, 0.0});
            }
          }
          --p;
          continue;
        }
        // The last negligible d[zero] in the block (none: there is none).
        std::size_t zero {none};
        for (std::size_t ks {p}; ks-- > top;) {
          const double around {std::abs(e[ks]) + (ks != top ? std::abs(e[ks - 1]) : 0.0)};
          if (std::abs(d[ks]) <= tiny + (epsilon * around)) {
            d[ks] = 0.0;
            zero  = ks;
            break;
          }
        }
        if (zero == p - 1) { // chase e[p−2] up the block with rotations on the right
          double f {e[p - 2]};
          e[p - 2] = 0.0;
          for (std::size_t j {p - 1}; j-- > top;) {
            const double t  {rotation_length(d[j], f)};
            const double cs {d[j] / t};
            const double sn {f / t};
            d[j] = t;
            if (j != top) {
              f        = -sn * e[j - 1];
              e[j - 1] = cs * e[j - 1];
            }
            if (record) {
              on_vt.push_back({row_operation::kind::rotate, j, p - 1, cs, sn});
            }
          }
          continue;
        }
        if (zero != none) { // split at d[zero] = 0: chase e[zero] along the row with rotations on the left
          double f {e[zero]};
          e[zero] = 0.0;
          for (std::size_t j {zero + 1}; j < p; ++j) {
            const double t  {rotation_length(d[j], f)};
            const double cs {d[j] / t};
            const double sn {f / t};
            d[j] = t;
            f    = -sn * e[j];
            e[j] = cs * e[j];
            if (record) {
              on_ut.push_back({row_operation::kind::rotate, j, zero, cs, sn});
            }
          }
          continue;
        }
        if (steps == max_steps) {
          throw convergence_error("SVD: the bidiagonal QR iteration did not converge within its steps");
        }
        ++steps;
        if (top > chased_last || p - 1 < chased_top) { // a new block: chase from its larger end
          down = std::abs(d[top]) >= std::abs(d[p - 1]);
        }
        chased_top  = top;
        chased_last = p - 1;
        if (down) {
          qr_step(top, p, [&](std::size_t position) -> double& { return d[position]; },
                  [&](std::size_t position) -> double& { return e[position]; }, [](std::size_t position) { return position; }, on_vt, on_ut);
        }
        else {
          // The step on the block turned end for end (J Bᵀ J, upper bidiagonal again): its rotations on the right are
          // B's on the left at the mirrored rows, and the other way round.
          const std::size_t ends {top + p - 1};
          qr_step(top, p, [&](std::size_t position) -> double& { return d[ends - position]; },
                  [&](std::size_t position) -> double& { return e[ends - position - 1]; },
                  [ends](std::size_t position) { return ends - position; },
                  on_ut, on_vt);
        }
      }
      if (!vt.empty()) {
        apply_row_operations(vt, n, on_vt);
        apply_row_operations(ut, n, on_ut);
      }
    }

    /*!
     * \brief The \ref power_of_two_scale of the largest magnitude of \p entry over \p rows rows of \p cols: by bands of rows on the
     *        threads, each band's largest then the largest of those (a maximum, the same whatever the order).
     */
    template <typename Entry>
    [[nodiscard]] double largest_magnitude(std::size_t  rows,
                                           std::size_t  cols,
                                           const Entry& entry)
    {
      const std::size_t   bands {(rows + svd_band - 1) / svd_band};
      std::vector<double> largest(bands, 0.0);
      parallel_for(bands, memory_element_weight * svd_band * cols, [&](std::size_t band) {
                     double most {0.0};
                     for (std::size_t i {band * svd_band}; i < std::min(rows, (band + 1) * svd_band); ++i) {
                       for (std::size_t j {0}; j < cols; ++j) {
                         most = std::max(most, static_cast<double>(std::abs(entry(i, j))));
                       }
                     }
                     largest[band] = most;
                   });
      const double scale {std::accumulate(largest.begin(), largest.end(), 0.0, [](double x, double y) { return std::max(x, y); })};
      return power_of_two_scale(scale);
    }

    //! \brief The conjugate transpose of \p a, as \ref numerical::transpose copies (by blocks, on the threads).
    template <typename T>
    [[nodiscard]] matrix<T> adjoint(const matrix<T>& a)
    {
      const std::size_t rows  {a.rows()};
      const std::size_t cols  {a.cols()};
      buffer<T>         entries(rows * cols);
      const std::size_t piece {transpose_piece(cols * sizeof(T), rows * sizeof(T))};
      parallel_for((cols + transpose_tile - 1) / transpose_tile, transpose_tile * rows, [&](std::size_t block) {
                     const std::size_t j0 {block * transpose_tile};
                     const std::size_t j1 {std::min(j0 + transpose_tile, cols)};
                     for (std::size_t i0 {0}; i0 < rows; i0 += transpose_tile) {
                       transpose_walk(j0, j1, i0, std::min(i0 + transpose_tile, rows), piece,
                                      [&](std::size_t j, std::size_t i) { entries[(j * rows) + i] = conj_of(a(i, j)); });
                     }
                   });
      return matrix<T> {cols, rows, std::move(entries)};
    }

    /*!
     * \brief Singular value \p i of the arrow `M` whose first row is `z` and whose diagonal is `d` (`d₀ = 0`, `d`
     *        strictly ascending): root `σ` of `1 + Σⱼ zⱼ²/(dⱼ² − σ²) = 0`, as LAPACK's `dlasd4`.
     *
     * Solved by \ref solve_secular in `μ = σ² − d_o²`, the poles' offsets `(dⱼ − d_o)(dⱼ + d_o)` computed from their
     * two accurate factors, so that both `dⱼ − σ` and `dⱼ + σ` come out with small relative errors.
     * \param[in]  d              The diagonal (the poles).
     * \param[in]  z              The first row (the weights).
     * \param[in]  i              The singular value.
     * \param[in]  max_iterations The bound on the steps.
     * \param[out] below          `dⱼ − σ` for every `j`.
     * \param[out] above          `dⱼ + σ` for every `j`.
     * \return `σ`.
     */
    [[nodiscard]] inline double singular_secular_root(const buffer<double>&      d,
                                                      const buffer<double>&      z,
                                                      std::size_t                i,
                                                      std::size_t                max_iterations,
                                                      double*                    below,
                                                      double*                    above)
    {
      buffer<double>         delta(d.size(), double {});
      const secular_solution root                   {solve_secular(z, 1.0, i, max_iterations, delta.data(), [&d](std::size_t o, std::size_t j) {
                                                                     return (d[j] - d[o]) * (d[j] + d[o]);
                                                                   })};
      const double           pole                   {d[root.origin]};
      const double           shift                  {root.offset / (pole + std::sqrt((pole * pole) + root.offset))}; // σ − d_o from σ² − d_o²
      for (std::size_t j {0}; j < d.size(); ++j) {
        below[j] = (d[j] - pole) - shift;
        above[j] = (d[j] + pole) + shift;
      }
      return pole + shift;
    }

    //! \brief Columns \p a and \p b of the row-major \p m (\p cols per row) rotated: `a ← c a + s b`, `b ← c b − s a`.
    inline void rotate_columns(buffer<double>&      m,
                               std::size_t          cols,
                               std::size_t          a,
                               std::size_t          b,
                               double               c,
                               double               s)
    {
      for (std::size_t r {0}; r < m.size() / cols; ++r) {
        const double x {m[(r * cols) + a]};
        const double y {m[(r * cols) + b]};
        m[(r * cols) + a] = (c * x) + (s * y);
        m[(r * cols) + b] = (c * y) - (s * x);
      }
    }

    //! \brief The rows a task of \ref merge_arrow reorders in place, one row's buffer between them.
    inline constexpr std::size_t sort_rows {64};

    //! \brief The buffers \ref merge_arrow works in, kept by each thread from one join to the next (\ref
    //!        thread_arrow_workspace): mapping their pages anew took 11–16 % of an `svd` of order 64 to 512 on one
    //!        x86-64 core, 13–20 % on six.
    struct arrow_workspace
    {
      workspace_buffer below    {}; //!< `poles[j] − roots[i]`, row by root; then their product with `above`, then the left vectors.
      workspace_buffer above    {}; //!< `poles[j] + roots[i]`; then the right vectors.
      workspace_buffer new_u    {}; //!< The kept columns of `u` times the left vectors.
      workspace_buffer new_v    {}; //!< The kept columns of `v` times the right vectors.
      workspace_buffer columns  {}; //!< One segment's rows of the kept columns that reach it.
      workspace_buffer factor   {}; //!< The rows of the arrow's vectors for those columns.
    };

    /*!
     * \brief The calling thread's \ref arrow_workspace, kept from one decomposition to the next. A join uses it whole
     *        on one thread and nests no other join; \ref release_large_arrow_workspace lets a large buffer go after.
     */
    [[nodiscard]] inline arrow_workspace& thread_arrow_workspace()
    {
      thread_local arrow_workspace work;
      return work;
    }

    //! \brief Releases the buffers of \p work longer than \ref kept_workspace.
    inline void release_large_arrow_workspace(arrow_workspace& work)
    {
      for (workspace_buffer* buffer : {&work.below, &work.above, &work.new_u, &work.new_v,
                                       &work.columns, &work.factor}) {
        if (buffer->size > kept_workspace) {
          buffer->data.reset();
          buffer->size = 0;
        }
      }
    }

    /*!
     * \brief The singular value decomposition of the arrow `M` (first row `z`, diagonal `d`, `d₀ = 0`), carried into
     *        the bases \p u (`n × n`) and \p v (`m × m`, its first `n` columns the arrow's; a last, null column
     *        when `m = n + 1`), as LAPACK's `dlasd1`–`dlasd3`: on return \p d holds the singular values ascending
     *        and \p u, \p v the singular vectors, \p v's null column last.
     *
     * Deflation first (`dlasd2`): a negligible `zⱼ` leaves `dⱼ` and its vectors as they are, a `dⱼ` near zero is
     * rotated into the zero pole, and two nearly equal `dⱼ` are rotated together so that one of them deflates.
     * The remaining roots are found by \ref singular_secular_root, in parallel; `z` is recomputed from them
     * (Löwner's formula), so that the vectors are orthogonal to working precision, and they reach \p u and \p v by
     * a matrix product each.
     * \param[in,out] d          The diagonal; on return, the singular values.
     * \param[in]     z          The first row.
     * \param[in]     alpha,beta The entries the arrow was built from: they set the deflation tolerance.
     * \param[in,out] u          The left basis; on return, the left singular vectors.
     * \param[in,out] v          The right basis; on return, the right singular vectors.
     * \param[in]     n          The arrow's order.
     * \param[in]     m          The right basis's order (`n`, or `n + 1` with a null column).
     * \param[in]     split      The row of \p u the arrow's first row is (its upper half above, its lower below):
     *                           the bases are block diagonal by it, and each half's rows take only the columns
     *                           that reach them (`dlasd3` multiplies the halves apart the same way).
     * \param[in,out] work       The buffers it works in, the caller's thread's (\ref thread_arrow_workspace).
     */
    inline void merge_arrow(buffer<double>&      d,
                            buffer<double>       z,
                            double               alpha,
                            double               beta,
                            buffer<double>&      u,
                            buffer<double>&      v,
                            std::size_t          n,
                            std::size_t          m,
                            std::size_t          split,
                            arrow_workspace&     work)
    {
      // The rows each column reaches: 1 the upper half's, 2 u's row `split` (the arrow's first), 4 the lower half's.
      // A rotation of two columns makes each reach the rows of both.
      std::vector<unsigned> reach_u(n, 4U);
      std::vector<unsigned> reach_v(m, 4U);
      reach_u[0] = 2U;
      reach_v[0] = m > n ? 5U : 1U; // the upper half's null vector, turned with the lower one's when there is one
      for (std::size_t j {1}; j <= split; ++j) {
        reach_u[j] = 1U;
        reach_v[j] = 1U;
      }
      const auto turned {[](std::vector<unsigned>& reach, std::size_t a, std::size_t b) {
                           reach[a] = reach[b] = reach[a] | reach[b];
                         }};
      std::vector<std::size_t> order(n - 1);
      for (std::size_t j {1}; j < n; ++j) {
        order[j - 1] = j;
      }
      std::sort(order.begin(), order.end(), [&d](std::size_t x, std::size_t y) { return d[x] < d[y]; });
      double largest {std::max(std::abs(alpha), std::abs(beta))};
      for (const double value : d) {
        largest = std::max(largest, value);
      }
      if (largest == 0.0) {
        return; // a zero arrow: its singular values are zeros, its vectors the bases as they are
      }
      const double tolerance {8.0 * std::numeric_limits<double>::epsilon() * largest};
      if (std::abs(z[0]) <= tolerance) {
        z[0] = tolerance; // the zero pole keeps a weight: the arrow moves by no more than the tolerance
      }
      std::vector<std::size_t> kept    {0}; // by ascending d, the zero pole first
      std::vector<std::size_t> deflated;    // the columns that keep their singular value
      std::size_t              pending {n}; // the last column kept so far, still open to a rotation with the next
      for (const std::size_t j : order) {
        if (std::abs(z[j]) <= tolerance) {
          deflated.push_back(j);
          continue;
        }
        if (d[j] <= tolerance) {
          // Nearly the zero pole: its weight joins the zero pole's, and dⱼ deflates.
          const double length {std::hypot(z[0], z[j])};
          rotate_columns(v, m, 0, j, z[0] / length, z[j] / length);
          turned(reach_v, 0, j);
          z[0] = length;
          z[j] = 0.0;
          deflated.push_back(j);
          continue;
        }
        if (pending != n && d[j] - d[pending] <= tolerance) {
          // Close values: one rotation of both rows and columns zeroes the pending one's weight.
          const double length {std::hypot(z[j], z[pending])};
          const double c      {z[j] / length};
          const double s      {z[pending] / length};
          rotate_columns(v, m, j, pending, c, s);
          rotate_columns(u, n, j, pending, c, s);
          turned(reach_v, j, pending);
          turned(reach_u, j, pending);
          z[j]       = length;
          z[pending] = 0.0;
          deflated.push_back(pending);
        }
        else if (pending != n) {
          kept.push_back(pending);
        }
        pending = j;
      }
      if (pending != n) {
        kept.push_back(pending);
      }
      const std::size_t   k                 {kept.size()};
      buffer<double>      poles(k, double   {});
      buffer<double>      weights(k, double {});
      for (std::size_t i {0}; i < k; ++i) {
        poles[i]   = d[kept[i]];
        weights[i] = z[kept[i]];
      }
      buffer<double> roots(k, double     {});
      double* const  below               {workspace_of(work.below, k * k)}; // below[i * k + j] = poles[j] − roots[i]
      double* const  above               {workspace_of(work.above, k * k)}; // above[i * k + j] = poles[j] + roots[i]
      parallel_for(k, 8 * k, [&](std::size_t i) {
                     roots[i] = singular_secular_root(poles, weights, i, secular_iterations, below + (i * k), above + (i * k));
                   });
      // below[i * k + j] ← (poles[j] − roots[i]) (poles[j] + roots[i]), the product each later use takes: above is free.
      parallel_for(k, k, [&](std::size_t i) {
                     for (std::size_t j {0}; j < k; ++j) {
                       below[(i * k) + j] *= above[(i * k) + j];
                     }
                   });
      double* const squares {below}; // squares[i * k + j] = poles[j]² − roots[i]²
      // Löwner: ẑⱼ² = Πᵢ (σᵢ² − dⱼ²) / Πᵢ≠ⱼ (dᵢ² − dⱼ²), each pole paired with its root, by blocks of poles on every
      // core, each pole's product in ascending order of the roots (a row of the gaps at a time, along the block).
      buffer<double> signed_weights(k, double {});
      parallel_for((k + lowner_block - 1) / lowner_block, lowner_block * k, [&](std::size_t block) {
                     const std::size_t                first {block * lowner_block};
                     const std::size_t                last {std::min(first + lowner_block, k)};
                     std::array<double, lowner_block> product {};
                     for (std::size_t j {first}; j < last; ++j) {
                       product[j - first] = -squares[(j * k) + j];
                     }
                     for (std::size_t i {0}; i < k; ++i) {
                       for (std::size_t j {first}; j < last; ++j) {
                         if (i != j) {
                           product[j - first] *= -squares[(i * k) + j] / ((poles[i] - poles[j]) * (poles[i] + poles[j]));
                         }
                       }
                     }
                     for (std::size_t j {first}; j < last; ++j) {
                       signed_weights[j] = std::copysign(std::sqrt(product[j - first]), weights[j]);
                     }
                   });
      // The vectors by rows of poles, each on its own core: right[j][i] = ẑⱼ / (dⱼ² − σᵢ²), read a column of the squares,
      // into above's storage; then the left ones, poles[j] times the right, into the squares' (column i: the vectors of
      // roots[i], in the kept columns' basis).
      double* const right {above};
      parallel_for(k, 4 * k, [&](std::size_t j) {
                     for (std::size_t i {0}; i < k; ++i) {
                       right[(j * k) + i] = signed_weights[j] / squares[(i * k) + j]; // ẑⱼ / (dⱼ² − σᵢ²)
                     }
                   });
      double* const left {below};
      parallel_for(k, k, [&](std::size_t j) {
                     for (std::size_t i {0}; i < k; ++i) {
                       left[(j * k) + i] = j == 0 ? -1.0 : poles[j] * right[(j * k) + i];
                     }
                   });
      for (double* const vectors : {left, right}) {
        buffer<double> length(k, 0.0);
        for (std::size_t j {0}; j < k; ++j) { // by rows, each a contiguous update of every column's sum
          for (std::size_t i {0}; i < k; ++i) {
            length[i] += vectors[(j * k) + i] * vectors[(j * k) + i];
          }
        }
        for (double& l : length) {
          l = std::sqrt(l);
        }
        parallel_for(k, k, [&](std::size_t j) {
                       for (std::size_t i {0}; i < k; ++i) {
                         vectors[(j * k) + i] /= length[i];
                       }
                     });
      }
      // The kept columns times the arrow's vectors; the deflated columns as they are; then by ascending value.
      // By segments of rows, each taking the kept columns that reach it and their rows of the arrow's vectors: the
      // others are zero there, and an element's sum without them is the same to the bit.
      using segment = std::tuple<std::size_t, std::size_t, unsigned>; // first row, rows, the bit of reach
      const auto carried {[&](const buffer<double>& basis, std::size_t rows, const double* small,
                              const std::vector<unsigned>& reach, const std::vector<segment>& segments, double* product) {
                            for (const auto& [first, count, bit] : segments) {
                              std::vector<std::size_t> reaching;
                              for (std::size_t i {0}; i < k; ++i) {
                                if ((reach[kept[i]] & bit) != 0U) {
                                  reaching.push_back(i);
                                }
                              }
                              const std::size_t   width {reaching.size()};
                              double* const       columns {workspace_of(work.columns, count * width)};
                              double* const       factor {workspace_of(work.factor, width * k)};
                              parallel_for(count, width + k, [&](std::size_t r) {
                                             for (std::size_t c {0}; c < width; ++c) {
                                               columns[(r * width) + c] = basis[((first + r) * rows) + kept[reaching[c]]];
                                             }
                                             std::fill_n(product + ((first + r) * k), k, 0.0); // the product adds to it
                                           });
                              for (std::size_t c {0}; c < width; ++c) {
                                std::copy_n(small + (reaching[c] * k), k, factor + (c * k));
                              }
                              gemm(count, k, width, columns, width, factor, k, product + (first * k), k);
                            }
                          }};
      double* const new_u {workspace_of(work.new_u, n * k)};
      double* const new_v {workspace_of(work.new_v, m * k)};
      carried(u, n, left, reach_u, {{0, split, 1U}, {split, 1, 2U}, {split + 1, n - split - 1, 4U}}, new_u);
      carried(v, m, right, reach_v, {{0, split + 1, 1U}, {split + 1, m - split - 1, 4U}}, new_v);
      buffer<double>            values(n, double {});
      std::vector<std::size_t>  from(n); // where each value's vectors are: a kept root (< k) or a deflated column
      for (std::size_t i {0}; i < k; ++i) {
        values[i] = roots[i];
        from[i]   = i;
      }
      for (std::size_t i {0}; i < deflated.size(); ++i) {
        values[k + i] = d[deflated[i]];
        from[k + i]   = k + i;
      }
      std::vector<std::size_t> rank(n);
      for (std::size_t j {0}; j < n; ++j) {
        rank[j] = j;
      }
      std::sort(rank.begin(), rank.end(), [&values](std::size_t x, std::size_t y) { return values[x] < values[y]; });
      std::vector<std::size_t> source(n); // by sorted column: a kept root (< k) or a deflated column (k + its place)
      for (std::size_t c {0}; c < n; ++c) {
        source[c] = from[rank[c]];
        d[c]      = values[rank[c]];
      }
      // Row by row on every core, in place: a row of the result reads only that row of the old basis (and of the
      // product), so each is gathered into a row of its own, then written over it (by columns, each write was a row
      // apart; a copy of each basis held one matrix more apiece). The right basis's null column, if any, stays last.
      parallel_for((m + sort_rows - 1) / sort_rows, sort_rows * n, [&](std::size_t block) {
                     buffer<double> row(n); // written whole for each row
                     for (std::size_t r {block * sort_rows}; r < std::min(m, (block + 1) * sort_rows); ++r) {
                       if (r < n) { // m ≥ n: the right basis has the rows of both
                         for (std::size_t c {0}; c < n; ++c) {
                           row[c] = source[c] < k ? new_u[(r * k) + source[c]] : u[(r * n) + deflated[source[c] - k]];
                         }
                         std::copy(row.begin(), row.end(), u.begin() + static_cast<std::ptrdiff_t>(r * n));
                       }
                       for (std::size_t c {0}; c < n; ++c) {
                         row[c] = source[c] < k ? new_v[(r * k) + source[c]] : v[(r * m) + deflated[source[c] - k]];
                       }
                       std::copy(row.begin(), row.end(), v.begin() + static_cast<std::ptrdiff_t>(r * m));
                     }
                   });
    }

    //! \brief The order up to which \ref bidiagonal_divide decomposes by implicit-shift QR.
    inline constexpr std::size_t bidiagonal_leaf {32};

    /*!
     * \brief One block of \ref bidiagonal_divide's tree: the bidiagonal it decomposes (its own copies scaled by a
     *        power of two, or its parent's), the two halves it splits into, and its decomposition once solved.
     */
    struct bidiagonal_node
    {
      const double*  d        {};  //!< The diagonal it is solved on.
      const double*  e        {};  //!< The superdiagonal it is solved on.
      std::size_t    n        {};  //!< Its rows.
      std::size_t    extra    {};  //!< Its columns past its rows, 0 or 1.
      int            exponent {0}; //!< The power of two it is scaled down by.
      buffer<double> scaled_d {};  //!< The scaled diagonal, when \p exponent is not 0.
      buffer<double> scaled_e {};  //!< The scaled superdiagonal, when \p exponent is not 0.
      std::size_t    k        {0}; //!< The row it splits at; 0 for a leaf.
      std::size_t    first    {};  //!< The block above row \p k.
      std::size_t    second   {};  //!< The block below row \p k.
      std::size_t    depth    {};  //!< Its distance from the root.
      buffer<double> sigma    {};  //!< Its singular values, ascending.
      buffer<double> u        {};  //!< Its left singular vectors, as columns.
      buffer<double> v        {};  //!< Its right singular vectors, as columns.
    };

    /*!
     * \brief Appends the block (\p d, \p e, \p n, \p extra) and everything below it to \p nodes, parents before
     *        children, each scaled to work near unit size by a power of two (exactly): the secular equation works on
     *        squares, which overflow past 1e154.
     * \return Its index in \p nodes.
     */
    NUMERICAL_OUT_OF_LINE inline std::size_t plan_bidiagonal(std::vector<bidiagonal_node>& nodes,
                                                             const double*                 d,
                                                             const double*                 e,
                                                             std::size_t                   n,
                                                             std::size_t                   extra,
                                                             std::size_t                   depth)
    {
      double largest {0.0};
      for (std::size_t i {0}; i < n; ++i) {
        largest = std::max({largest, std::abs(d[i]), i + 1 < n + extra ? std::abs(e[i]) : 0.0});
      }
      bidiagonal_node node {.d = d, .e = e, .n = n, .extra = extra, .depth = depth};
      (void)std::frexp(largest, &node.exponent);
      if (node.exponent != 0) {
        node.scaled_d.assign(n, double {});
        node.scaled_e.assign(n, 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          node.scaled_d[i] = std::ldexp(d[i], -node.exponent);
          node.scaled_e[i] = i + 1 < n + extra ? std::ldexp(e[i], -node.exponent) : 0.0;
        }
        node.d = node.scaled_d.data(); // the buffers' storage stays put as the node moves into the vector
        node.e = node.scaled_e.data();
      }
      const std::size_t index {nodes.size()};
      if (n <= bidiagonal_leaf) {
        nodes.push_back(std::move(node));
        return index;
      }
      node.k = n / 2;
      const double* const work_d {node.d};
      const double* const work_e {node.e};
      const std::size_t   k      {node.k};
      nodes.push_back(std::move(node));
      const std::size_t first    {plan_bidiagonal(nodes, work_d, work_e, k, 1, depth + 1)};
      const std::size_t second   {plan_bidiagonal(nodes, work_d + k + 1, work_e + k + 1, n - k - 1, extra, depth + 1)};
      nodes[index].first  = first;
      nodes[index].second = second;
      return index;
    }

    //! \brief Its singular values and vectors into \p node's, from scratch: \p u zero, \p v the identity.
    NUMERICAL_OUT_OF_LINE inline void start_bidiagonal_node(bidiagonal_node& node)
    {
      const std::size_t m {node.n + node.extra};
      node.sigma.assign(node.n, 0.0);
      node.u.assign(node.n * node.n, 0.0);
      node.v.assign(m * m, 0.0);
      for (std::size_t i {0}; i < m; ++i) {
        node.v[(i * m) + i] = 1.0;
      }
    }

    //! \brief Decomposes the leaf \p node by implicit-shift QR.
    NUMERICAL_OUT_OF_LINE inline void solve_bidiagonal_leaf(bidiagonal_node& node)
    {
      start_bidiagonal_node(node);
      const double* const d     {node.d};
      const double* const e     {node.e};
      const std::size_t   n     {node.n};
      const std::size_t   extra {node.extra};
      const std::size_t   m     {n + extra};
      buffer<double>&     sigma {node.sigma};
      buffer<double>&     u     {node.u};
      buffer<double>&     v     {node.v};
      buffer<double>      diagonal(d, d + n);
      buffer<double>      super(n, 0.0);
      std::copy_n(e, n - std::min<std::size_t>(n, 1), super.begin());
      // An extra column first: rotations of it against each column, from the last row up, chase its one entry
      // out (B G = [B' 0]), so the square B' remains.
      double fill {extra == 1 ? e[n - 1] : 0.0};
      for (std::size_t j {n}; extra == 1 && j-- > 0;) {
        const double length {std::hypot(diagonal[j], fill)};
        const double c      {length == 0.0 ? 1.0 : diagonal[j] / length};
        const double s      {length == 0.0 ? 0.0 : fill / length};
        diagonal[j] = length;
        rotate_columns(v, m, j, n, c, s);
        if (j > 0) {
          fill         = -s * super[j - 1];
          super[j - 1] = c * super[j - 1];
        }
      }
      buffer<double> ut(n * n, 0.0);
      buffer<double> vt(n * n, 0.0);
      for (std::size_t i {0}; i < n; ++i) {
        ut[(i * n) + i] = 1.0;
        vt[(i * n) + i] = 1.0;
      }
      bidiagonal_qr(diagonal, super, ut, vt, n, svd_qr_steps_per_value * n);   // descending
      buffer<double> rotated(v);
      for (std::size_t c {0}; c < n; ++c) {
        const std::size_t source {n - 1 - c};
        sigma[c] = diagonal[source];
        for (std::size_t r {0}; r < n; ++r) {
          u[(r * n) + c] = ut[(source * n) + r];
        }
        for (std::size_t r {0}; r < m; ++r) {   // G times B''s right vector
          double sum {0.0};
          for (std::size_t t {0}; t < n; ++t) {
            sum += rotated[(r * m) + t] * vt[(source * n) + t];
          }
          v[(r * m) + c] = sum;
        }
      }
    }

    //! \brief Joins the two decomposed halves of \p node, \p one and \p two, by \ref merge_arrow, \p node started
    //!        (\ref start_bidiagonal_node).
    NUMERICAL_OUT_OF_LINE inline void join_bidiagonal_node(bidiagonal_node&       node,
                                                           const bidiagonal_node& one,
                                                           const bidiagonal_node& two,
                                                           arrow_workspace&       work)
    {
      const std::size_t     n      {node.n};
      const std::size_t     extra  {node.extra};
      const std::size_t     m      {n + extra};
      const std::size_t     k      {node.k};
      const std::size_t     lower  {n - k - 1};
      const double          alpha  {node.d[k]};
      const double          beta   {node.e[k]};
      const buffer<double>& sigma1 {one.sigma};
      const buffer<double>& u1     {one.u};
      const buffer<double>& v1     {one.v};
      const buffer<double>& sigma2 {two.sigma};
      const buffer<double>& u2     {two.u};
      const buffer<double>& v2     {two.v};
      buffer<double>&       u      {node.u};
      buffer<double>&       v      {node.v};
      const std::size_t     m1     {k + 1};
      const std::size_t     m2     {lower + extra};
      buffer<double>        arrow(n, 0.0);
      buffer<double>        z(n, 0.0);
      // The bases: row k is the arrow's first row; the arrow's columns are the null vector, then the halves'.
      u[k * n] = 1.0;
      for (std::size_t c {0}; c < k; ++c) {
        arrow[1 + c] = sigma1[c];
        z[1 + c]     = alpha * v1[(k * m1) + c];
        for (std::size_t r {0}; r < k; ++r) {
          u[(r * n) + 1 + c] = u1[(r * k) + c];
        }
      }
      for (std::size_t c {0}; c < lower; ++c) {
        arrow[k + 1 + c] = sigma2[c];
        z[k + 1 + c]     = beta * v2[c];
        for (std::size_t r {0}; r < lower; ++r) {
          u[((k + 1 + r) * n) + k + 1 + c] = u2[(r * lower) + c];
        }
      }
      std::fill(v.begin(), v.end(), 0.0);
      for (std::size_t r {0}; r < m1; ++r) {
        v[(r * m) + 0] = v1[(r * m1) + k];
        for (std::size_t c {0}; c < k; ++c) {
          v[(r * m) + 1 + c] = v1[(r * m1) + c];
        }
      }
      for (std::size_t r {0}; r < m2; ++r) {
        for (std::size_t c {0}; c < lower; ++c) {
          v[((m1 + r) * m) + k + 1 + c] = v2[(r * m2) + c];
        }
      }
      z[0] = alpha * v1[(k * m1) + k];
      if (extra == 1) { // the two null vectors: one joins the arrow, the other is the whole's
        for (std::size_t r {0}; r < m2; ++r) {
          v[((m1 + r) * m) + n] = v2[(r * m2) + lower];
        }
        const double other  {beta * v2[lower]};
        const double length {std::hypot(z[0], other)};
        const double c      {length == 0.0 ? 1.0 : z[0] / length};
        const double s      {length == 0.0 ? 0.0 : other / length};
        rotate_columns(v, m, 0, n, c, s);
        z[0] = length;
      }
      merge_arrow(arrow, std::move(z), alpha, beta, u, v, n, m, k, work);
      node.sigma = std::move(arrow);
    }

    //! \brief The singular values of a solved \p node scaled back.
    NUMERICAL_OUT_OF_LINE inline void unscale_bidiagonal_node(bidiagonal_node& node)
    {
      if (node.exponent != 0) {
        for (double& value : node.sigma) {
          value = std::ldexp(value, node.exponent);
        }
      }
    }

    //! \brief Joins the halves of \p nodes' block \p index in \p work, then lets them go.
    NUMERICAL_OUT_OF_LINE inline void join_bidiagonal_halves(std::vector<bidiagonal_node>& nodes,
                                                             std::size_t                   index,
                                                             arrow_workspace&              work)
    {
      bidiagonal_node& node {nodes[index]};
      join_bidiagonal_node(node, nodes[node.first], nodes[node.second], work);
      unscale_bidiagonal_node(node);
      nodes[node.first]  = bidiagonal_node {}; // read by no other join
      nodes[node.second] = bidiagonal_node {};
    }

    //! \brief Decomposes the block \p index of \p nodes and everything below it, each block right after its halves.
    NUMERICAL_OUT_OF_LINE inline void solve_bidiagonal_subtree(std::vector<bidiagonal_node>& nodes,
                                                               std::size_t                   index,
                                                               arrow_workspace&              work)
    {
      if (nodes[index].k == 0) {
        solve_bidiagonal_leaf(nodes[index]);
        unscale_bidiagonal_node(nodes[index]);
        return;
      }
      start_bidiagonal_node(nodes[index]); // before its halves, as the recursion did: their buffers then come back to it
      solve_bidiagonal_subtree(nodes, nodes[index].first, work);
      solve_bidiagonal_subtree(nodes, nodes[index].second, work);
      join_bidiagonal_halves(nodes, index, work);
    }

    /*!
     * \brief The singular value decomposition of the upper bidiagonal `n × (n + extra)` (`extra` 0 or 1) with
     *        diagonal `d[0…n-1]` and superdiagonal `e[0…n+extra-2]` (`e[n-1]` the last column's entry when `extra`
     *        is 1), by divide and conquer, as LAPACK's `dbdsdc`: \p sigma ascending, \p u (`n × n`) and \p v
     *        (`m × m`, `m = n + extra`, the null vector last) row-major, their columns the singular vectors.
     *
     * The rows split at a middle row `k`: above it an `k × (k + 1)` bidiagonal, below it an `(n − k − 1) × (…)` one,
     * each decomposed the same way (by implicit-shift QR up to \ref bidiagonal_leaf); row `k`'s two entries, in
     * their right bases, with the two null vectors rotated into one, make an arrow, which \ref merge_arrow
     * decomposes.
     *
     * The tree is laid out first (\ref plan_bidiagonal). The blocks at the depth that gives each thread one, and the
     * leaves above it, are independent of one another and go to the threads together, each decomposed on one
     * thread, a block right after its halves while they are still in its caches; the joins above them then come a
     * level at a time, those of one level together, a level of one join sharing it among the threads. Every block
     * takes the same inputs either way, so the result is the same to the bit; on one thread the order is the
     * recursion's.
     */
    inline void bidiagonal_divide(const double*        d,
                                  const double*        e,
                                  std::size_t          n,
                                  std::size_t          extra,
                                  buffer<double>&      sigma,
                                  buffer<double>&      u,
                                  buffer<double>&      v)
    {
      std::vector<bidiagonal_node> nodes;
      (void)plan_bidiagonal(nodes, d, e, n, extra, 0);
      const auto               cut {static_cast<std::size_t>(std::bit_width(max_threads() - 1))};
      std::vector<std::size_t> tasks;
      for (std::size_t i {0}; i < nodes.size(); ++i) {
        if (nodes[i].depth == cut || (nodes[i].depth < cut && nodes[i].k == 0)) {
          tasks.push_back(i);
        }
      }
      const std::size_t task_order {nodes[tasks[0]].n};
      parallel_for(tasks.size(), tasks.size() > 1 ? task_order * task_order * task_order : 0, [&](std::size_t t) {
                     solve_bidiagonal_subtree(nodes, tasks[t], thread_arrow_workspace());
                     release_large_arrow_workspace(thread_arrow_workspace());
                   });
      for (std::size_t level {cut}; level-- > 0;) {
        std::vector<std::size_t> joins;
        for (std::size_t i {0}; i < nodes.size(); ++i) {
          if (nodes[i].k != 0 && nodes[i].depth == level) {
            joins.push_back(i);
          }
        }
        const std::size_t order {joins.empty() ? 0 : nodes[joins[0]].n};
        parallel_for(joins.size(), joins.size() > 1 ? order * order * order / 4 : 0, [&](std::size_t j) {
                       start_bidiagonal_node(nodes[joins[j]]);
                       join_bidiagonal_halves(nodes, joins[j], thread_arrow_workspace());
                       release_large_arrow_workspace(thread_arrow_workspace());
                     });
      }
      sigma = std::move(nodes[0].sigma);
      u     = std::move(nodes[0].u);
      v     = std::move(nodes[0].v);
    }

    /*!
     * \brief The singular values of the upper bidiagonal (\p d, \p e) into \p d, descending, the singular vectors
     *        accumulated into the rows of \p ut and \p vt as \ref bidiagonal_qr accumulates them, by divide and
     *        conquer (\ref bidiagonal_divide): most of the work in matrix products.
     * \throws convergence_error If an iteration did not converge within its bound.
     */
    template <typename T>
    void bidiagonal_divide_and_conquer(buffer<double>&      d,
                                       buffer<double>&      e,
                                       buffer<T>&           ut,
                                       buffer<T>&           vt,
                                       std::size_t          n)
    {
      if (n == 0) {
        return;
      }
      buffer<double> sigma;
      buffer<double> u;
      buffer<double> v;
      bidiagonal_divide(d.data(), e.data(), n, 0, sigma, u, v);
      buffer<double> left(n * n, double  {}); // row c: the vectors of the c-th largest value
      buffer<double> right(n * n, double {});
      for (std::size_t c {0}; c < n; ++c) {
        d[c] = sigma[n - 1 - c];
        for (std::size_t r {0}; r < n; ++r) {
          left[(c * n) + r]  = u[(r * n) + (n - 1 - c)];
          right[(c * n) + r] = v[(r * n) + (n - 1 - c)];
        }
      }
      multiply_real_left(left, ut, n);
      multiply_real_left(right, vt, n);
    }
  } // namespace detail

  //! \brief How \ref numerical::svd_decomposition computes the decomposition of `R`, after a Householder QR.
  enum class svd_method
  {
    bidiagonal_qr,     //!< Golub–Kahan bidiagonalisation then implicit-shift QR, as LAPACK's `dgesvd`.
    jacobi,            //!< One-sided Jacobi: several times slower, each singular value to high relative accuracy.
    divide_and_conquer //!< Golub–Kahan bidiagonalisation then divide and conquer, as LAPACK's `dgesdd`: the default.
  };

  /*!
   * \brief The singular value decomposition `A = U Σ Vᴴ` of an `m × n` matrix.
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   *
   * With `k = min(m, n)`: the `k` singular values, descending; `U` (`m × k`) and `V` (`n × k`) with
   * orthonormal columns, also where `A` is rank-deficient. Computed after a Householder QR of `A` (of `Aᴴ`
   * when `m < n`): by default `R` is reduced to bidiagonal form and the bidiagonal decomposed by divide and
   * conquer (\ref svd_method::divide_and_conquer), the method of LAPACK's `dgesdd`, most of its work in matrix
   * products, or by implicit-shift QR (\ref svd_method::bidiagonal_qr, `dgesvd`'s); either way each singular
   * value to an absolute accuracy of the largest times the machine epsilon. One-sided Jacobi on `R`
   * (\ref svd_method::jacobi) gives every singular value to high relative accuracy — the small ones of a
   * graded matrix too — at several times the cost. \ref vh is `Vᴴ`, the third factor NumPy's `svd` returns.
   * Full (`full` true, NumPy's `full_matrices=True`), `U` is `m × m` and `V` `n × n`: the columns past `k` are an
   * orthonormal basis of the complement, the rest of the complete Householder `Q` of the larger side.
   */
  template <field T>
  class svd_decomposition
  {
  public:

    /*!
     * \brief Decomposes \p a.
     * \param[in] a      The matrix.
     * \param[in] method The algorithm for `R`.
     * \param[in] full   Whether `U` and `V` are square (`m × m`, `n × n`), as NumPy's `full_matrices=True`.
     * \throws non_finite_error If an entry is infinite or NaN.
     * \throws convergence_error If the iteration does not converge within its bound (not observed).
     */
    explicit svd_decomposition(const matrix<T>& a,
                               svd_method       method = svd_method::divide_and_conquer,
                               bool             full   = false)
    {
      detail::require_finite(a, "SVD");
      if (a.rows() >= a.cols()) {
        factor(a, method, full);
      }
      else { // A = (Aᴴ)ᴴ = (U' Σ V'ᴴ)ᴴ = V' Σ U'ᴴ
        factor(detail::adjoint(a), method, full);
        std::swap(u_, v_);
      }
    }

    /*!
     * \brief Decomposes \p a in its own storage, an expiring matrix's elements taken over: its scaled copy is
     *        made in place, one matrix less at the peak.
     * \param[in] a      The matrix.
     * \param[in] method The algorithm for `R`.
     * \param[in] full   Whether `U` and `V` are square (`m × m`, `n × n`), as NumPy's `full_matrices=True`.
     * \throws non_finite_error If an entry is infinite or NaN.
     * \throws convergence_error If the iteration does not converge within its bound (not observed).
     */
    explicit svd_decomposition(matrix<T>&& a,
                               svd_method  method = svd_method::divide_and_conquer,
                               bool        full   = false)
    {
      detail::require_finite(a, "SVD");
      if (a.rows() >= a.cols()) {
        factor(std::move(a), method, full);
      }
      else { // as the constructor of a constant matrix
        factor(detail::adjoint(a), method, full);
        std::swap(u_, v_);
      }
    }

    //! \brief The singular values, descending.
    [[nodiscard]] const vector<double>& singular_values() const noexcept
    {
      return singular_values_;
    }

    //! \brief The left singular vectors `U` (columns): `m × k`, or `m × m` full.
    [[nodiscard]] const matrix<T>& u() const& noexcept
    {
      return u_;
    }

    //! \brief `U`, moved out of an expiring decomposition (\ref v and the singular values stay).
    [[nodiscard]] matrix<T> u() &&
    {
      return std::move(u_);
    }

    //! \brief The right singular vectors `V` (columns): `n × k`, or `n × n` full.
    [[nodiscard]] const matrix<T>& v() const& noexcept
    {
      return v_;
    }

    //! \brief `Vᴴ` (`k × n`, or `n × n` full), so that `A = U Σ Vᴴ` reads `u() * diag(singular_values()) * vh()` (full,
    //!        `Σ` padded with zeros to `m × n`).
    [[nodiscard]] matrix<T> vh() const
    {
      return detail::adjoint(v_);
    }

  private:

    /*!
     * \brief `U = Q [D L; 0]` (`rows × width`): the left vectors \p left of `R` (`cols × cols`), each row times its
     *        diagonal's phase \p phases (`D`, as \ref detail::householder_qr rotates `Q`), and where `U` is square the
     *        identity past them, `Q`'s columns past `cols`; `Q` the product of the reflectors \p blocks (\ref detail::apply_q).
     */
    [[nodiscard]] static matrix<T> applied_left_vectors(const buffer<T>&                               left,
                                                        const std::vector<detail::reflector_block<T>>& blocks,
                                                        const std::vector<T>&                          phases,
                                                        std::size_t                                    rows,
                                                        std::size_t                                    cols,
                                                        std::size_t                                    width)
    {
      buffer<T> u(rows * width, T {});
      for (std::size_t i {0}; i < cols; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          u[(i * width) + j] = phases[i] * left[(i * cols) + j];
        }
      }
      for (std::size_t i {cols}; i < width; ++i) {
        u[(i * width) + i] = T {1};
      }
      detail::apply_q(u, blocks, rows, width);
      return matrix<T> {rows, width, std::move(u)};
    }

    /*!
     * \brief `U` from the Householder \p q (`rows × width`) and the left vectors \p left of `R` (`cols × cols`): `Q`
     *        times them, and when `Q` is complete (`width` = `rows`) its columns past `cols` after them.
     */
    [[nodiscard]] static matrix<T> left_vectors(buffer<T>   q,
                                                buffer<T>   left,
                                                std::size_t rows,
                                                std::size_t cols,
                                                std::size_t width)
    {
      if (width == cols) {
        return matrix<T> {rows, cols, std::move(q)} *matrix<T> {cols, cols, std::move(left)};
      }
      buffer<T> u(rows * rows, T {});
      detail::gemm(rows, cols, cols, q.data(), rows, left.data(), cols, u.data(), rows);
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {cols}; j < rows; ++j) {
          u[(i * rows) + j] = q[(i * rows) + j];
        }
      }
      return matrix<T> {rows, rows, std::move(u)};
    }

    //! \brief Decomposes an `m × n` \p a with `m ≥ n` into \ref u_, \ref singular_values_ and \ref v_, `U` square when
    //!        \p full.
    void factor(const matrix<T>& a,
                svd_method       method,
                bool             full)
    {
      factor_scaled(std::to_address(a.begin()), buffer<T>(a.rows() * a.cols()), a.rows(), a.cols(), method, full);
    }

    //! \brief \ref factor of an expiring matrix, its elements scaled in place.
    void factor(matrix<T>&& a,
                svd_method  method,
                bool        full)
    {
      const std::size_t rows    {a.rows()};
      const std::size_t cols    {a.cols()};
      buffer<T>         entries {std::move(a).elements()};
      const T* const    source  {entries.data()};
      factor_scaled(source, std::move(entries), rows, cols, method, full);
    }

    /*!
     * \brief The decomposition of the `rows × cols` matrix at \p source (row-major), its scaled copy written into
     *        \p entries, which may hold the source itself (each element read before it is written).
     */
    void factor_scaled(const T*    source,
                       buffer<T>   entries,
                       std::size_t rows,
                       std::size_t cols,
                       svd_method  method,
                       bool        full)
    {
      // Work on A divided by its largest magnitude: the sums of squares in the sweeps overflow near
      // 1e155 and underflow near 1e-155. The singular values are scaled back; the vectors do not change.
      // Both passes by bands of rows on the threads, bound by the memory: a band's largest magnitude, then the
      // largest of those (exact in any order), then the quotients.
      const std::size_t   bands  {(rows + detail::svd_band - 1) / detail::svd_band};
      std::vector<double> largest(bands, 0.0);
      detail::parallel_for(bands, detail::memory_element_weight * detail::svd_band * cols, [source, rows, cols, &largest](std::size_t band) {
                             double most {0.0};
                             for (std::size_t e {band * detail::svd_band * cols}; e < std::min(rows, (band + 1) * detail::svd_band) * cols; ++e) {
                               most = std::max(most, static_cast<double>(std::abs(source[e])));
                             }
                             largest[band] = most;
                           });
      // the largest over no band (no row) is 0, as over zeros
      double scale              {std::accumulate(largest.begin(), largest.end(), 0.0, [](double x, double y) { return std::max(x, y); })};
      scale = detail::power_of_two_scale(scale);
      T* const          into    {entries.data()};
      const T           divisor {detail::from_carried<T>(scale)};
      detail::parallel_for(bands, detail::memory_element_weight * detail::svd_band * cols, [source, into, rows, cols, divisor](std::size_t band) {
                             for (std::size_t e {band * detail::svd_band * cols}; e < std::min(rows, (band + 1) * detail::svd_band) * cols; ++e) {
                               into[e] = source[e] / divisor;
                             }
                           });
      // A square matrix is reduced to bidiagonal form as it is (as LAPACK's dgesdd does short of a tall one): a QR
      // first, its Q formed and multiplied in at the end, took a third of the decomposition's time at order 1024.
      const bool         direct  {rows == cols && method != svd_method::jacobi};
      const std::size_t  width   {full ? rows : cols};
      // A tall matrix's U is Q times R's, taken through Q's reflectors (applied), where Q formed then multiplied in took
      // as much work again and held a matrix more (ten AVX-512 cores, one thread, 1000 × 100: 9.8 → 8.4 ms).
      const bool                              applied {!direct && method != svd_method::jacobi};
      std::vector<detail::reflector_block<T>> blocks;
      std::vector<T>                          phases;
      detail::thin_qr<T>                      factors;
      if (direct) {
        factors.r = std::move(entries);
      }
      else if (applied) {
        detail::qr_reflections<T> reflections {detail::householder_reflect(std::move(entries), rows, cols)};
        blocks    = std::move(reflections.blocks);
        factors.r = std::move(reflections.r);
        phases.resize(cols);
        for (std::size_t k {0}; k < cols; ++k) { // R's diagonal real and non-negative, as householder_qr makes it
          phases[k] = detail::phase_of(factors.r[(k * cols) + k]);
          for (std::size_t j {k}; j < cols; ++j) {
            factors.r[(k * cols) + j] *= detail::conj_of(phases[k]);
          }
        }
      }
      else {
        factors = detail::householder_qr(std::move(entries), rows, cols, full);
      }
      if (method != svd_method::jacobi) {
        buffer<double> d(cols, double {});
        buffer<double> e(cols, double {});
        buffer<T>      ut;
        buffer<T>      vt;
        detail::bidiagonalize(factors.r, cols, d, e, ut, vt);
        factors.r = buffer<T> {}; // read no more: its storage back before the vectors' updates
        if (method == svd_method::bidiagonal_qr) {
          detail::bidiagonal_qr(d, e, ut, vt, cols, detail::svd_qr_steps_per_value * cols);
        }
        else {
          detail::bidiagonal_divide_and_conquer(d, e, ut, vt, cols);
        }
        buffer<T> left(cols * cols); // rows of ut and vt are the vectors: made columns, each freed once copied
        detail::transpose_into(ut.data(), cols, cols, left.data());
        ut = buffer<T> {};
        buffer<T> right(cols * cols);
        detail::transpose_into(vt.data(), cols, cols, right.data());
        vt = buffer<T> {};
        for (double& value : d) {
          value *= scale;
        }
        singular_values_ = vector<double> {std::move(d)};
        u_               = direct ? matrix<T> {cols, cols, std::move(left)} : applied_left_vectors(left, blocks, phases, rows, cols, width);
        v_               = matrix<T> {cols, cols, std::move(right)};
        return;
      }
      buffer<T>     w                    {std::move(factors.r)}; // A = Q R, and R V = W with orthogonal columns
      buffer<T>     right(cols * cols, T {});
      for (std::size_t i {0}; i < cols; ++i) {
        right[(i * cols) + i] = T {1};
      }
      detail::one_sided_jacobi(w, right, cols, detail::jacobi_sweeps);

      buffer<double> norms(cols, double {});
      for (std::size_t j {0}; j < cols; ++j) {
        buffer<T> column(cols, T {});
        for (std::size_t i {0}; i < cols; ++i) {
          column[i] = w[(i * cols) + j];
        }
        norms[j] = static_cast<double>(detail::scaled_norm(column));
      }
      std::vector<std::size_t> order(cols);
      for (std::size_t j {0}; j < cols; ++j) {
        order[j] = j;
      }
      std::stable_sort(order.begin(), order.end(), [&norms](std::size_t x, std::size_t y) { return norms[x] > norms[y]; });

      // The left vectors of R: each nonzero column of W over its norm; a zero one has none, and takes
      // the unit vector that leaves the largest residual against the columns already set, made
      // orthogonal to them.
      buffer<T>      left(cols * cols, T         {});
      buffer<T>      sorted_right(cols * cols, T {});
      buffer<double> values(cols, double         {});
      for (std::size_t d {0}; d < cols; ++d) {
        const std::size_t j {order[d]};
        values[d] = norms[j] * scale;
        for (std::size_t i {0}; i < cols; ++i) {
          sorted_right[(i * cols) + d] = right[(i * cols) + j];
        }
        if (norms[j] > 0.0) {
          for (std::size_t i {0}; i < cols; ++i) {
            left[(i * cols) + d] = w[(i * cols) + j] / detail::from_carried<T>(norms[j]);
          }
          continue;
        }
        buffer<T>      best;
        double         best_norm {-1.0};
        for (std::size_t k {0}; k < cols; ++k) {
          buffer<T> candidate(cols, T {});
          candidate[k] = T {1};
          orthogonalise(candidate, left, cols, d);
          const auto norm {static_cast<double>(detail::scaled_norm(candidate))};
          if (norm > best_norm) {
            best      = std::move(candidate);
            best_norm = norm;
          }
        }
        orthogonalise(best, left, cols, d); // a second pass restores orthogonality lost to rounding
        const auto norm {static_cast<double>(detail::scaled_norm(best))};
        for (std::size_t i {0}; i < cols; ++i) {
          left[(i * cols) + d] = best[i] / detail::from_carried<T>(norm);
        }
      }
      singular_values_ = vector<double> {std::move(values)};
      u_               = left_vectors(std::move(factors.q), std::move(left), rows, cols, width);
      v_               = matrix<T> {cols, cols, std::move(sorted_right)};
    }

    //! \brief Removes from \p x its components along the first \p count columns of the `n × n` \p basis.
    static void orthogonalise(buffer<T>&            x,
                              const buffer<T>&      basis,
                              std::size_t           n,
                              std::size_t           count)
    {
      for (std::size_t c {0}; c < count; ++c) {
        T projection {};
        for (std::size_t i {0}; i < n; ++i) {
          projection += detail::conj_of(basis[(i * n) + c]) * x[i];
        }
        for (std::size_t i {0}; i < n; ++i) {
          x[i] -= projection * basis[(i * n) + c];
        }
      }
    }

    vector<double> singular_values_; //!< Singular values, descending.
    matrix<T>      u_;               //!< m×k left singular vectors.
    matrix<T>      v_;               //!< n×k right singular vectors.
  };

  namespace detail {

    //! \brief How many of the singular values \p sigma of an `rows × cols` matrix exceed \p cutoff: its numerical
    //!        rank (the first that many, where \p sigma is descending).
    [[nodiscard]] inline std::size_t numerical_rank(const vector<double>& sigma,
                                                    std::size_t           rows,
                                                    std::size_t           cols,
                                                    rank_cutoff           cutoff = {})
    {
      double largest {0.0};
      for (std::size_t d {0}; d < sigma.size(); ++d) {
        largest = std::max(largest, sigma[d]);
      }
      const double tolerance {cutoff.magnitude(largest, rows, cols)};
      std::size_t  rank      {0};
      for (std::size_t d {0}; d < sigma.size(); ++d) {
        rank += static_cast<std::size_t>(sigma[d] > tolerance);
      }
      return rank;
    }

    //! \brief How far above the rank cutoff a bound on the smallest singular value must stay for \ref lstsq_by_qr to
    //!        answer: nearer, the singular value decomposition decides the rank.
    inline constexpr double qr_rank_margin {100.0};

    //! \brief \p x times two to the \p exponent, exactly where the result is a normal number (no power of two is
    //!        formed, so none overflows).
    template <typename T>
    [[nodiscard]] T scaled_by_two(const T& x,
                                  int      exponent)
    {
      return std::scalbn(x, exponent);
    }

    //! \brief \ref scaled_by_two for a complex number, part by part.
    template <typename U>
    [[nodiscard]] std::complex<U> scaled_by_two(const std::complex<U>& x,
                                                int                    exponent)
    {
      return std::complex<U> {std::scalbn(x.real(), exponent), std::scalbn(x.imag(), exponent)};
    }

    /*!
     * \brief `R⁻¹`, upper triangular, of the `n × n` upper-triangular \p r (row-major): back substitution against
     *        the identity in parallel chunks of columns, each row's terms in registers (\ref subtract_products). A
     *        chunk's rows past its last column stay zero; a zero on the diagonal leaves infinities or NaN.
     */
    template <typename T>
    [[nodiscard]] buffer<T> upper_inverse(const buffer<T>&      r,
                                          std::size_t           n)
    {
      constexpr std::size_t width      {64};
      buffer<T>             x(n * n, T {});
      parallel_for((n + width - 1) / width, n * n * width / 2, [&](std::size_t chunk) {
                     const std::size_t first {chunk * width};
                     const std::size_t last {std::min(first + width, n)};
                     for (std::size_t i {last}; i-- > 0;) {
                       T* const row {x.data() + (i * n) + first};
                       if (i >= first) {
                         row[i - first] = T {1};
                       }
                       subtract_products(row, r.data() + (i * n) + i + 1, x.data() + ((i + 1) * n) + first, n, last - (i + 1), last - first);
                       for (std::size_t c {0}; c < last - first; ++c) {
                         row[c] = row[c] / r[(i * n) + i];
                       }
                     }
                   });
      return x;
    }

    /*!
     * \brief The least-squares solution of `A X ≈ B` by a QR factorisation, of `A`, or of `Aᴴ` for a wide `A` (the
     *        minimum-norm solution), where it is the one the singular value decomposition gives: with `A` divided by
     *        a power of two near its largest magnitude, the bound `1 / ‖R⁻¹‖_F` under its smallest singular value must
     *        stay \ref qr_rank_margin times above \p cutoff of the bound `‖R‖_F` over its largest, so that every
     *        singular value is far above the cutoff and the rank full (a zero on `R`'s diagonal makes the first bound
     *        zero, or NaN, which fails). Else nothing. A fraction of the decomposition's work, the rest in matrix
     *        products.
     * \param[in]  a      The `m × n` matrix `A`.
     * \param[in]  b      The right-hand sides, one per column.
     * \param[in]  cutoff Where a singular value counts as zero.
     * \param[out] r Where given, the square `R` factorised, of `A` divided by `2^exponent`, and that exponent: its
     *              singular values are `A`'s over `2^exponent`.
     * \return The solution, or nothing where the decomposition must decide the rank.
     */
    template <typename T>
    [[nodiscard]] std::optional<matrix<T>> lstsq_by_qr(const matrix<T>&                       a,
                                                       const matrix<T>&                       b,
                                                       rank_cutoff                            cutoff,
                                                       std::pair<buffer<T>, int>* const       r = nullptr)
    {
      const std::size_t m       {a.rows()};
      const std::size_t n       {a.cols()};
      const bool        wide    {m < n};
      const std::size_t rows    {wide ? n : m}; // of the matrix factorised
      const std::size_t cols    {wide ? m : n};
      double            largest {0.0};
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          largest = std::max(largest, static_cast<double>(std::abs(a(i, j))));
        }
      }
      if (largest == 0.0) {
        return std::nullopt;
      }
      const int      exponent               {std::ilogb(largest)};
      buffer<T>      entries(rows * cols, T {});
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          if (wide) {
            entries[(j * cols) + i] = scaled_by_two(conj_of(a(i, j)), -exponent);
          }
          else {
            entries[(i * cols) + j] = scaled_by_two(a(i, j), -exponent);
          }
        }
      }
      qr_reflections<T>       factors         {householder_reflect(std::move(entries), rows, cols)};
      const buffer<T>         inverse         {upper_inverse(factors.r, cols)};
      double                  r_squares       {0.0};
      double                  inverse_squares {0.0};
      for (std::size_t e {0}; e < cols * cols; ++e) {
        r_squares       += static_cast<double>(std::norm(factors.r[e]));
        inverse_squares += static_cast<double>(std::norm(inverse[e]));
      }
      const double smallest_bound {1.0 / std::sqrt(inverse_squares)};
      if (!(smallest_bound > qr_rank_margin * cutoff.scaled(exponent).magnitude(std::sqrt(r_squares), m, n))) {
        return std::nullopt;
      }
      const std::size_t k          {b.cols()};
      buffer<T>         x(n * k, T {});
      if (wide) { // x = Q [R⁻ᴴ B; 0]: the blocks applied last first
        const matrix<T>      inverse_adjoint {adjoint(matrix<T> {cols, cols, inverse})};
        const buffer<T>      right(b.begin(), b.end());
        gemm(cols, k, cols, &*inverse_adjoint.begin(), cols, right.data(), k, x.data(), k);
        for (std::size_t block {factors.blocks.size()}; block-- > 0;) {
          apply_reflector_block(factors.blocks[block], false, x, k, 0, k);
        }
      }
      else { // x = R⁻¹ times the first rows of Qᴴ B: the blocks' adjoints applied first first
        buffer<T> y(b.begin(), b.end());
        for (const reflector_block<T>& block : factors.blocks) {
          apply_reflector_block(block, true, y, k, 0, k);
        }
        gemm(n, k, cols, inverse.data(), cols, y.data(), k, x.data(), k);
      }
      for (T& value : x) {
        value = scaled_by_two(value, -exponent);
      }
      if (r != nullptr) {
        *r = {std::move(factors.r), exponent};
      }
      return matrix<T> {n, k, std::move(x)};
    }

    /*!
     * \brief The singular values of the square `n × n` row-major \p b, descending, without its vectors: those of its
     *        Golub–Kahan bidiagonal by dqds (\ref dqds_singular_values), no basis formed; by the implicit-shift QR
     *        iteration (\ref bidiagonal_qr) where dqds does not converge within \p sweeps steps per value.
     * \throws convergence_error If neither converges within its bound (not observed).
     */
    template <typename T>
    [[nodiscard]] buffer<double> square_singular_values(buffer<T>      b,
                                                        std::size_t    n,
                                                        qd_index       sweeps = dqds_sweeps)
    {
      buffer<double> d(n, double {});
      buffer<double> e(n, double {});
      buffer<T>      none;
      buffer<T>      none_too;
      bidiagonalize(b, n, d, e, none, none_too, false);
      if (!dqds_singular_values(d, e, n, sweeps)) {
        bidiagonal_qr(d, e, none, none_too, n, svd_qr_steps_per_value * n);
      }
      return d;
    }

    /*!
     * \brief The minimum-norm least-squares solution of `A X ≈ B` from the singular value decomposition \p s of `A`
     *        (`m × n`), its first \p rank triplets: `X = V Σ⁺ Uᴴ B`.
     */
    template <typename T>
    [[nodiscard]] matrix<T> lstsq_from_svd(const svd_decomposition<T>& s,
                                           std::size_t                 rank,
                                           const matrix<T>&            b)
    {
      // Through the matrix product kernel, as inverse_on_kept: the coefficients `Uᴴ B` (each row over its σ), then
      // `V` times them; summed term by term, each coefficient read U and B a column apart.
      const vector<double>& sigma {s.singular_values()};
      const std::size_t     rows  {s.u().rows()};
      const std::size_t     cols  {s.v().rows()};
      const std::size_t     sides {b.cols()};
      buffer<T>             left_adjoint(rank * rows);
      buffer<T>             right(cols * rank);
      for (std::size_t d {0}; d < rank; ++d) {
        for (std::size_t i {0}; i < rows; ++i) {
          left_adjoint[(d * rows) + i] = conj_of(s.u()(i, d));
        }
        for (std::size_t j {0}; j < cols; ++j) {
          right[(j * rank) + d] = s.v()(j, d);
        }
      }
      buffer<T> coefficients(rank * sides);
      gemm(rank, sides, rows, left_adjoint.data(), rows, std::to_address(b.begin()), sides, coefficients.data(), sides, c_start::zero);
      for (std::size_t d {0}; d < rank; ++d) {
        for (std::size_t c {0}; c < sides; ++c) {
          coefficients[(d * sides) + c] /= from_carried<T>(sigma[d]);
        }
      }
      buffer<T> x(cols * sides);
      gemm(cols, sides, rank, right.data(), rank, coefficients.data(), sides, x.data(), sides, c_start::zero);
      return matrix<T> {cols, sides, std::move(x)};
    }
  } // namespace detail

  /*!
   * \brief The minimum-norm least-squares solution of `A x ≈ b`, as NumPy's `lstsq` gives it.
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   *
   * From the SVD of `A`: `x = V Σ⁺ Uᴴ b`, where a singular value at or under \p cutoff (by default NumPy's,
   * \ref rank_cutoff) counts as zero. Any shape, any rank: an overdetermined full-rank system gives its least-squares
   * solution, an underdetermined or rank-deficient one the solution of smallest norm.
   * \param[in] a      The `m × n` matrix.
   * \param[in] b      The right-hand side (size `m`).
   * \param[in] cutoff Where a singular value counts as zero.
   * \return `x` (size `n`).
   * \throws dimension_error If `b`'s size differs from the row count.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <field T>
  [[nodiscard]] vector<T> lstsq(const matrix<T>& a,
                                const vector<T>& b,
                                rank_cutoff      cutoff = {})
  {
    const matrix<T> column {lstsq(a, matrix<T> {b.size(), 1, buffer<T>(b.begin(), b.end())}, cutoff)};
    return vector<T> {buffer<T>(column.begin(), column.end())};
  }

  /*!
   * \brief The minimum-norm least-squares solution of `A X ≈ B` for every column of `B` at once, as
   *        `numpy.linalg.lstsq` with a matrix right-hand side: one SVD of `A`, `X = V Σ⁺ Uᴴ B`; or, where a bound
   *        on the condition number shows the rank full with room to spare, the same solution to rounding by one
   *        QR factorisation (\ref detail::lstsq_by_qr).
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   * \param[in] a      The `m × n` matrix.
   * \param[in] b      The right-hand sides (`m × k`).
   * \param[in] cutoff Where a singular value counts as zero.
   * \return `X` (`n × k`).
   * \throws dimension_error If `b`'s row count differs from `a`'s.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <field T>
  [[nodiscard]] matrix<T> lstsq(const matrix<T>& a,
                                const matrix<T>& b,
                                rank_cutoff      cutoff = {})
  {
    if (b.rows() != a.rows()) {
      throw dimension_error("lstsq: right-hand side size differs from the matrix");
    }
    detail::require_finite(a, "lstsq");
    if (std::optional<matrix<T>> fast {detail::lstsq_by_qr(a, b, cutoff)}) {
      return *std::move(fast);
    }
    const svd_decomposition<T> s {a};
    return detail::lstsq_from_svd(s, detail::numerical_rank(s.singular_values(), a.rows(), a.cols(), cutoff), b);
  }

  /*!
   * \brief The singular values of \p a, descending, without its singular vectors, as
   *        `numpy.linalg.svd(a, compute_uv=False)`: the Householder `R` of a tall `A` (of `Aᴴ` when wide), its
   *        Golub–Kahan bidiagonal and the dqds algorithm on it (LAPACK's `dlasq1`, as `dgesdd` takes it without
   *        vectors), no basis formed. Each value to an absolute accuracy of the largest times the machine epsilon,
   *        as \ref svd_decomposition's; those of a bidiagonal `A` each to high relative accuracy.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   * \throws convergence_error If the iteration does not converge within its bound (not observed).
   */
  template <field T>
  [[nodiscard]] vector<double> singular_values(const matrix<T>& a)
  {
    detail::require_finite(a, "singular values");
    const bool        tall {a.rows() >= a.cols()};
    const std::size_t rows {tall ? a.rows() : a.cols()};
    const std::size_t cols {tall ? a.cols() : a.rows()};
    // Divided by the power of two of its largest magnitude, as the decomposition is: the sums of squares overflow near
    // 1e155. The largest found by bands of rows on the threads (a maximum, whatever the order); each entry divided
    // once, copied by bands of rows, or by blocks where the matrix is read transposed.
    const double scale   {detail::largest_magnitude(a.rows(), a.cols(), [&a](std::size_t i, std::size_t j) { return a(i, j); })};
    const T      divisor {detail::from_carried<T>(scale)};
    buffer<T>    entries(rows * cols);
    if (tall) {
      detail::parallel_for((rows + detail::svd_band - 1) / detail::svd_band, detail::memory_element_weight * detail::svd_band * cols, [&](std::size_t band) {
                             for (std::size_t i {band * detail::svd_band}; i < std::min(rows, (band + 1) * detail::svd_band); ++i) {
                               for (std::size_t j {0}; j < cols; ++j) {
                                 entries[(i * cols) + j] = a(i, j) / divisor;
                               }
                             }
                           });
    }
    else {
      const std::size_t piece {detail::transpose_piece(rows * sizeof(T), cols * sizeof(T))};
      detail::parallel_for((rows + transpose_tile - 1) / transpose_tile, transpose_tile * cols, [&](std::size_t block) {
                             const std::size_t i0 {block * transpose_tile};
                             const std::size_t i1 {std::min(i0 + transpose_tile, rows)};
                             for (std::size_t j0 {0}; j0 < cols; j0 += transpose_tile) {
                               detail::transpose_walk(i0, i1, j0, std::min(j0 + transpose_tile, cols), piece,
                                                      [&](std::size_t i, std::size_t j) { entries[(i * cols) + j] = detail::conj_of(a(j, i)) / divisor; });
                             }
                           });
    }
    buffer<T>      r {rows == cols ? std::move(entries) : detail::householder_reflect(std::move(entries), rows, cols, false).r};
    buffer<double> d {detail::square_singular_values(std::move(r), cols)};
    for (double& value : d) {
      value *= scale;
    }
    return vector<double> {std::move(d)};
  }

  /*!
   * \brief The eigenvalues of the real symmetric or complex Hermitian \p a, ascending, without its eigenvectors, as
   *        `numpy.linalg.eigvalsh`: its Householder tridiagonal (no basis formed) and the implicit QL iteration on it,
   *        no rotation kept. Only the lower triangle is read, and of the diagonal its real part. Each value to an
   *        absolute accuracy of the largest magnitude times the machine epsilon, as \ref symmetric_eigen's.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry read is infinite or NaN.
   * \throws convergence_error If the iteration does not converge within its bound (not observed).
   */
  template <field T>
  [[nodiscard]] vector<double> symmetric_eigenvalues(const matrix<T>& a)
  {
    const std::size_t n {a.rows()};
    if (a.cols() != n) {
      throw dimension_error("symmetric eigenvalues require a square matrix");
    }
    detail::require_finite_lower(a, "symmetric eigenvalues");
    // Divided by the power of two of its largest magnitude read, as the decompositions are: a sum of squares overflows
    // near 1e155. Found and divided by bands of rows on the threads, each row written whole (its upper part zero).
    const double scale {detail::largest_magnitude(n, n, [&a](std::size_t i, std::size_t j) {
                                                    return j < i ? a(i, j) : j == i ? detail::from_carried<T>(detail::real_part(a(i, i))) : T {};
                                                  })};
    buffer<T> entries(n * n);
    detail::parallel_for((n + detail::svd_band - 1) / detail::svd_band, detail::memory_element_weight * detail::svd_band * n, [&](std::size_t band) {
                           for (std::size_t i {band * detail::svd_band}; i < std::min(n, (band + 1) * detail::svd_band); ++i) {
                             for (std::size_t j {0}; j < i; ++j) {
                               entries[(i * n) + j] = a(i, j) / detail::from_carried<T>(scale);
                             }
                             entries[(i * n) + i] = detail::from_carried<T>(detail::real_part(a(i, i)) / scale);
                             std::fill(entries.begin() + static_cast<std::ptrdiff_t>((i * n) + i + 1), entries.begin() + static_cast<std::ptrdiff_t>((i + 1) * n), T {});
                           }
                         });
    buffer<double> d(n, double {});
    if (n > 0) {
      buffer<double> off(n, double {});
      buffer<double> none;
      static_cast<void>(detail::tridiagonalize(entries, n, d, off));
      detail::tridiagonal_ql(d, off, none, n, detail::ql_steps_per_value * n);
    }
    std::sort(d.begin(), d.end());
    for (double& value : d) {
      value *= scale;
    }
    return vector<double> {std::move(d)};
  }

  //! \brief A least-squares solution and what `numpy.linalg.lstsq` reports beside it (\ref least_squares).
  template <typename T>
  struct least_squares_solution
  {
    matrix<T>      solution;        //!< The minimum-norm `x` of `A x ≈ B` (`n × k`), \ref lstsq's.
    vector<double> residuals;       //!< Per column of `B`, `‖B − A x‖²`: for a tall `A` of full column rank, else empty.
    std::size_t    rank {0};        //!< The numerical rank of `A` (\ref matrix_rank).
    vector<double> singular_values; //!< The singular values of `A`, descending (\ref singular_values).
  };

  /*!
   * \brief \ref lstsq, with the residuals, the rank and the singular values `numpy.linalg.lstsq` returns beside the
   *        solution: the residual sums of squares are given only where the solution is unique and need not be exact,
   *        `A` tall (`m > n`) and of full column rank, as NumPy gives them.
   * \param[in] a      The `m × n` matrix.
   * \param[in] b      The right-hand sides (`m × k`).
   * \param[in] cutoff Where a singular value counts as zero, in the solution and the rank.
   * \throws dimension_error If \p b's rows differ from \p a's.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] least_squares_solution<T> least_squares(const matrix<T>& a,
                                                        const matrix<T>& b,
                                                        rank_cutoff      cutoff = {})
  {
    if (b.rows() != a.rows()) {
      throw dimension_error("lstsq: right-hand side size differs from the matrix");
    }
    detail::require_finite(a, "lstsq");
    // The singular values from what the solution took: the QR's R (A's over a power of two), else the decomposition.
    std::pair<buffer<T>, int>           r;
    std::optional<matrix<T>>            fast {detail::lstsq_by_qr(a, b, cutoff, &r)};
    std::optional<svd_decomposition<T>> s;
    if (!fast) {
      s.emplace(a);
    }
    buffer<double> values {fast ? detail::square_singular_values(std::move(r.first), std::min(a.rows(), a.cols()))
                                     : buffer<double>(s->singular_values().begin(), s->singular_values().end())};
    if (fast) {
      for (double& value : values) {
        value = std::ldexp(value, r.second);
      }
    }
    const vector<double> sigma {std::move(values)};
    const std::size_t    rank  {detail::numerical_rank(sigma, a.rows(), a.cols(), cutoff)};
    matrix<T>            x     {fast ? *std::move(fast) : detail::lstsq_from_svd(*s, rank, b)};
    buffer<double>       residuals;
    if (rank == a.cols() && a.rows() > a.cols()) {
      const matrix<T> fitted {a * x};
      residuals.assign(b.cols(), 0.0);
      for (std::size_t i {0}; i < b.rows(); ++i) {
        for (std::size_t c {0}; c < b.cols(); ++c) {
          residuals[c] += std::norm(b(i, c) - fitted(i, c)); // |·|², real or complex
        }
      }
    }
    return least_squares_solution<T> {.solution        = std::move(x),
                                      .residuals       = vector<double> {std::move(residuals)},
                                      .rank            = rank,
                                      .singular_values = sigma};
  }

  namespace detail {

    //! \brief The real symmetric or complex Hermitian matrix whose lower triangle is the square \p a's, and of its
    //!        diagonal the real part.
    template <typename T>
    [[nodiscard]] matrix<T> hermitian_from_lower(const matrix<T>& a)
    {
      const std::size_t n                {a.rows()};
      buffer<T>         entries(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < i; ++j) {
          entries[(i * n) + j] = a(i, j);
          entries[(j * n) + i] = conj_of(a(i, j));
        }
        entries[(i * n) + i] = from_carried<T>(real_part(a(i, i)));
      }
      return matrix<T> {n, n, std::move(entries)};
    }

    /*!
     * \brief `Σ_d right_d (1 / σ_d) left_dᴴ` over the columns `d` of \p right (`cols × k`) and \p left (`rows × k`)
     *        whose \p sigma is above \p cutoff (of the largest magnitude in \p sigma): a pseudo-inverse from a
     *        singular value decomposition (`right = V`, `left = U`) or from an eigendecomposition (`V` both, `σ` the
     *        eigenvalues, signed). Through the matrix product kernel: summed term by term it ran five times as long as
     *        NumPy's at order 512.
     */
    template <typename T>
    [[nodiscard]] matrix<T> inverse_on_kept(const vector<double>& sigma,
                                            const matrix<T>&      right,
                                            const matrix<T>&      left,
                                            rank_cutoff           cutoff)
    {
      const std::size_t rows    {left.rows()};
      const std::size_t cols    {right.rows()};
      double            largest {0.0};
      for (std::size_t d {0}; d < sigma.size(); ++d) {
        largest = std::max(largest, std::abs(sigma[d]));
      }
      const double             tolerance {cutoff.magnitude(largest, rows, cols)};
      std::vector<std::size_t> kept;
      for (std::size_t d {0}; d < sigma.size(); ++d) {
        if (std::abs(sigma[d]) > tolerance) {
          kept.push_back(d);
        }
      }
      const std::size_t rank                        {kept.size()};
      buffer<T>         scaled(cols * rank, T       {});
      buffer<T>         left_adjoint(rank * rows, T {});
      for (std::size_t e {0}; e < rank; ++e) {
        const std::size_t d       {kept[e]};
        const auto        inverse {from_carried<T>(1.0 / sigma[d])};
        for (std::size_t i {0}; i < cols; ++i) {
          scaled[(i * rank) + e] = right(i, d) * inverse;
        }
        for (std::size_t j {0}; j < rows; ++j) {
          left_adjoint[(e * rows) + j] = conj_of(left(j, d));
        }
      }
      buffer<T> out(cols * rows, T {});
      gemm(cols, rows, rank, scaled.data(), rank, left_adjoint.data(), rows, out.data(), rows);
      return matrix<T> {cols, rows, std::move(out)};
    }

    /*!
     * \brief The singular values of the real symmetric or complex Hermitian matrix whose lower triangle is \p a's: of
     *        the matrix made whole, for a scalar the eigendecompositions do not take (they compute in `double`).
     */
    template <typename T>
    [[nodiscard]] vector<double> hermitian_singular_values(const matrix<T>& a)
    {
      return singular_values(hermitian_from_lower(a));
    }

    //! \brief The magnitudes of the eigenvalues of the real symmetric or complex Hermitian \p a, its lower triangle read:
    //!        its singular values, unordered.
    template <typename T>
    [[nodiscard]] vector<double> eigenvalue_magnitudes(const matrix<T>& a)
    {
      const vector<double> signed_values {symmetric_eigenvalues(a)};
      buffer<double>       values(signed_values.begin(), signed_values.end());
      for (double& value : values) {
        value = std::abs(value);
      }
      return vector<double> {std::move(values)};
    }

    //! \brief \ref hermitian_singular_values in `double`: the magnitudes of the eigenvalues, cheaper.
    [[nodiscard]] inline vector<double> hermitian_singular_values(const matrix<double>& a)
    {
      return eigenvalue_magnitudes(a);
    }

    //! \brief \ref hermitian_singular_values in `std::complex<double>`: the magnitudes of the eigenvalues, cheaper.
    [[nodiscard]] inline vector<double> hermitian_singular_values(const matrix<std::complex<double>>& a)
    {
      return eigenvalue_magnitudes(a);
    }

    /*!
     * \brief The pseudo-inverse of the real symmetric or complex Hermitian matrix whose lower triangle is \p a's: from
     *        the singular value decomposition of the matrix made whole, for a scalar the eigendecompositions do not take.
     */
    template <typename T>
    [[nodiscard]] matrix<T> hermitian_pseudo_inverse(const matrix<T>& a,
                                                     rank_cutoff      cutoff)
    {
      const svd_decomposition<T> s {hermitian_from_lower(a)};
      return inverse_on_kept(s.singular_values(), s.v(), s.u(), cutoff);
    }

    //! \brief \ref hermitian_pseudo_inverse in `double`: `V Λ⁺ Vᵀ` from the eigendecomposition, cheaper.
    [[nodiscard]] inline matrix<double> hermitian_pseudo_inverse(const matrix<double>& a,
                                                                 rank_cutoff           cutoff)
    {
      const symmetric_eigen decomposition {a};
      const matrix<double>& vectors       {decomposition.vectors()};
      return inverse_on_kept(decomposition.values(), vectors, vectors, cutoff);
    }

    //! \brief \ref hermitian_pseudo_inverse in `std::complex<double>`: `V Λ⁺ Vᴴ` from the eigendecomposition, cheaper.
    [[nodiscard]] inline matrix<std::complex<double>> hermitian_pseudo_inverse(const matrix<std::complex<double>>& a,
                                                                               rank_cutoff                         cutoff)
    {
      const hermitian_eigen               decomposition {a};
      const matrix<std::complex<double>>& vectors       {decomposition.vectors()};
      return inverse_on_kept(decomposition.values(), vectors, vectors, cutoff);
    }
  } // namespace detail

  /*!
   * \brief The numerical rank of \p a, as `numpy.linalg.matrix_rank` counts it: its singular values above \p cutoff.
   * \param[in] a         The matrix.
   * \param[in] cutoff    Where a singular value counts as zero; by default NumPy's (\ref rank_cutoff).
   * \param[in] hermitian Whether \p a is real symmetric or complex Hermitian, as NumPy's `hermitian=True`: its
   *                      singular values are then the magnitudes of its eigenvalues, cheaper to find (in `double`,
   *                      real or complex; any other scalar takes its singular values), and only its lower triangle is
   *                      read.
   * \throws dimension_error  If \p hermitian and \p a is not square.
   * \throws non_finite_error If an entry read is infinite or NaN.
   */
  template <field T>
  [[nodiscard]] std::size_t matrix_rank(const matrix<T>& a,
                                        rank_cutoff      cutoff    = {},
                                        bool             hermitian = false)
  {
    if (hermitian) {
      if (a.cols() != a.rows()) {
        throw dimension_error("matrix_rank: a Hermitian matrix is square");
      }
      detail::require_finite_lower(a, "matrix_rank");
      return detail::numerical_rank(detail::hermitian_singular_values(a), a.rows(), a.cols(), cutoff);
    }
    return detail::numerical_rank(singular_values(a), a.rows(), a.cols(), cutoff);
  }

  /*!
   * \brief The Moore–Penrose pseudo-inverse of the `m × n` \p a (`n × m`), as `numpy.linalg.pinv` gives it:
   *        `V Σ⁺ Uᴴ`, a singular value at or under \p cutoff counting as zero.
   * \param[in] a         The matrix.
   * \param[in] cutoff    Where a singular value counts as zero; by default NumPy's (\ref rank_cutoff).
   * \param[in] hermitian Whether \p a is real symmetric or complex Hermitian, as NumPy's `hermitian=True`: from its
   *                      eigendecomposition `V Λ Vᴴ`, `V Λ⁺ Vᴴ`, cheaper (in `double`, real or complex; any other
   *                      scalar takes its singular value decomposition), and only its lower triangle is read.
   * \throws dimension_error  If \p hermitian and \p a is not square.
   * \throws non_finite_error If an entry read is infinite or NaN.
   */
  template <field T>
  [[nodiscard]] matrix<T> pinv(const matrix<T>& a,
                               rank_cutoff      cutoff    = {},
                               bool             hermitian = false)
  {
    if (hermitian) {
      if (a.cols() != a.rows()) {
        throw dimension_error("pinv: a Hermitian matrix is square");
      }
      detail::require_finite_lower(a, "pinv");
      return detail::hermitian_pseudo_inverse(a, cutoff);
    }
    detail::require_finite(a, "pinv");
    const svd_decomposition<T> s {a};
    return detail::inverse_on_kept(s.singular_values(), s.v(), s.u(), cutoff);
  }
} // namespace numerical

#endif // NUMERICAL_SVD_HPP
