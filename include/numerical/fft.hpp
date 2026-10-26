/*!
 * \file fft.hpp
 * \brief The discrete Fourier transform of a tensor along an axis, as `numpy.fft.fft` and `numpy.fft.ifft` compute it.
 *
 * A length that is a power of two runs Stockham passes of radix 4 (and one of radix 2 where the exponent is odd), each
 * reading and writing in order; any other length the transform's sum itself, term after term, until its own algorithm
 * lands. The twiddle factors are computed in double-double arithmetic by one fixed sequence of operations, each a sum
 * or an explicit fused multiply-add that no compiler may contract otherwise, then rounded once: the same factors on
 * every machine, whatever its `sin` and `cos`. A butterfly's products are rounded apart from its sums, so a transform
 * has the same bits on every path, any number of threads, alone or among others.
 */
#ifndef NUMERICAL_FFT_HPP
#define NUMERICAL_FFT_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "buffer.hpp"
#include "gemm.hpp"
#include "parallel.hpp"
#include "simd.hpp"
#include "tensor.hpp"
#include "tensor_ops.hpp"

namespace numerical {

  //! \brief How a transform and its inverse are scaled, as NumPy's `norm`.
  enum class fft_norm : unsigned char
  {
    backward, //!< The forward transform unscaled, the inverse scaled by 1/n (NumPy's default).
    ortho,    //!< Both scaled by 1/√n: a unitary pair.
    forward,  //!< The forward transform scaled by 1/n, the inverse unscaled.
  };

  namespace detail {

    //! \brief A double-double number: `hi + lo`, `hi` the sum rounded to `double`.
    struct double_double
    {
      double hi; //!< The leading part.
      double lo; //!< The trailing part, at most half a unit in the last place of `hi`.
    };

    //! \brief `a + b` exactly, as a sum and its rounding error (Knuth's two-sum).
    [[nodiscard]] inline double_double two_sum(double a,
                                               double b) noexcept
    {
      const double sum      {a + b};
      const double b_part   {sum - a};
      const double a_part   {sum - b_part};
      const double b_error  {b - b_part};
      const double a_error  {a - a_part};
      return {sum, a_error + b_error};
    }

    //! \brief `a + b` exactly where `|a| ≥ |b|` or `a` is 0 (Dekker's fast two-sum).
    [[nodiscard]] inline double_double fast_two_sum(double a,
                                                    double b) noexcept
    {
      const double sum   {a + b};
      const double b_out {sum - a};
      return {sum, b - b_out};
    }

    //! \brief The sum of two double-double numbers.
    [[nodiscard]] inline double_double dd_add(double_double x,
                                              double_double y) noexcept
    {
      const double_double high  {two_sum(x.hi, y.hi)};
      const double_double low   {two_sum(x.lo, y.lo)};
      double              error {high.lo + low.hi};
      const double_double part  {fast_two_sum(high.hi, error)};
      error = part.lo + low.lo;
      return fast_two_sum(part.hi, error);
    }

    //! \brief The product of two double-double numbers, its cross terms by explicit fused multiply-adds.
    [[nodiscard]] inline double_double dd_mul(double_double x,
                                              double_double y) noexcept
    {
      const double product {x.hi * y.hi};
      double       error   {std::fma(x.hi, y.hi, -product)};
      error = std::fma(x.hi, y.lo, error);
      error = std::fma(x.lo, y.hi, error);
      return fast_two_sum(product, error);
    }

    //! \brief \p x divided by the integer \p d (exact in a `double`).
    [[nodiscard]] inline double_double dd_div(double_double x,
                                              double        d) noexcept
    {
      const double first     {x.hi / d};
      double       remainder {std::fma(-first, d, x.hi)}; // exact: the quotient's rounding error times d
      remainder = remainder + x.lo;
      return fast_two_sum(first, remainder / d);
    }

    //! \brief π/4 as a double-double number.
    inline constexpr double_double quarter_pi {0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55};

    //! \brief The terms of \ref dd_sin_cos' series: 1/k! for k = 0 to 29, each from the last by \ref dd_div.
    [[nodiscard]] inline const std::array<double_double, 30>& inverse_factorials()
    {
      static const std::array<double_double, 30> table {[] {
                                                          std::array<double_double, 30> terms {};
                                                          terms[0] = {1.0, 0.0};
                                                          for (std::size_t k {1}; k < terms.size(); ++k) {
                                                            terms[k] = dd_div(terms[k - 1], static_cast<double>(k));
                                                          }
                                                          return terms;
                                                        }()};
      return table;
    }

    /*!
     * \brief The cosine and sine of \p t, in [0, π/4], as double-double numbers: their Taylor series in `t²` by Horner's
     *        rule to the term of degree 28 and 29, whose next terms are under 10⁻³² of the sum there.
     */
    [[nodiscard]] inline std::pair<double_double, double_double> dd_sin_cos(double_double t)
    {
      const std::array<double_double, 30>& terms  {inverse_factorials()};
      const double_double                  square {dd_mul(t, t)};
      double_double                        cosine {terms[28]};
      double_double                        sine   {terms[29]};
      for (std::size_t k {28}; k >= 2; k -= 2) { // cos: Σ (−1)ᵏ t²ᵏ/(2k)!, sin: t · Σ (−1)ᵏ t²ᵏ/(2k + 1)!
        const double_double c           {terms[k - 2]};
        const double_double s           {terms[k - 1]};
        const double_double cosine_step {dd_mul(cosine, square)};
        const double_double sine_step   {dd_mul(sine, square)};
        cosine = dd_add(c, {-cosine_step.hi, -cosine_step.lo});
        sine   = dd_add(s, {-sine_step.hi, -sine_step.lo});
      }
      return {cosine, dd_mul(sine, t)};
    }

    //! \brief \p x with a zero's sign made positive: the twiddle factors' zeros are +0, as their exact values.
    [[nodiscard]] inline double unsigned_zero(double x) noexcept
    {
      return x == 0.0 ? 0.0 : x;
    }

    //! \brief \p x negated, both parts.
    [[nodiscard]] inline double_double dd_negate(double_double x) noexcept
    {
      return {-x.hi, -x.lo};
    }

    //! \brief `e^{−2πi k/n}`'s angle reduced to an eighth of a turn by integer arithmetic, `8k = octant·n + rest`: the
    //!        octant, and the angle's share of π/4 times n, `rest` (`n − rest` in an odd octant, measured back from its
    //!        end). \p n is at most 2⁵⁰, so that `8k` and the share are exact.
    struct octant_share
    {
      std::size_t octant; //!< The eighth of a turn the angle falls in.
      std::size_t share;  //!< The angle within it, in units of π/(4n).
    };

    //! \brief The \ref octant_share of `e^{−2πi k/n}`.
    [[nodiscard]] inline octant_share reduce_to_octant(std::size_t k,
                                                       std::size_t n) noexcept
    {
      const std::size_t eighths {8 * (k % n)};
      const std::size_t octant  {eighths / n};
      const std::size_t rest    {eighths % n};
      return {octant, (octant % 2) == 1 ? n - rest : rest};
    }

    //! \brief `e^{−2πi k/n}` from the cosine \p c and sine \p s of its angle within the octant \p octant (\ref
    //!        reduce_to_octant): as double-double parts, real then imaginary.
    [[nodiscard]] inline std::pair<double_double, double_double> place_by_octant(std::size_t   octant,
                                                                                 double_double c,
                                                                                 double_double s) noexcept
    {
      double_double cosine {c};
      double_double sine   {s};
      switch (octant) { // the angle (π/4)·(octant + rest/n): within an odd octant, measured back from its end
        case 1: cosine = s; sine = c; break;
        case 2: cosine = dd_negate(s); sine = c; break;
        case 3: cosine = dd_negate(c); sine = s; break;
        case 4: cosine = dd_negate(c); sine = dd_negate(s); break;
        case 5: cosine = dd_negate(s); sine = dd_negate(c); break;
        case 6: cosine = s; sine = dd_negate(c); break;
        case 7: cosine = c; sine = dd_negate(s); break;
        default: break; // the first octant as computed
      }
      return {cosine, dd_negate(sine)};
    }

    //! \brief The cosine and sine of `(π/4)·share/n` by \ref dd_sin_cos.
    [[nodiscard]] inline std::pair<double_double, double_double> octant_sin_cos(std::size_t share,
                                                                                std::size_t n)
    {
      return dd_sin_cos(dd_mul(quarter_pi, dd_div({static_cast<double>(share), 0.0}, static_cast<double>(n))));
    }

    //! \brief `e^{−2πi k/n}` as double-double parts: \ref reduce_to_octant, \ref octant_sin_cos, \ref place_by_octant.
    [[nodiscard]] inline std::pair<double_double, double_double> unit_root_parts(std::size_t k,
                                                                                 std::size_t n)
    {
      const octant_share reduced {reduce_to_octant(k, n)};
      const auto [c, s] {octant_sin_cos(reduced.share, n)};
      return place_by_octant(reduced.octant, c, s);
    }

    //! \brief \ref unit_root_parts rounded once to `double`, a zero's sign positive as the exact value's.
    [[nodiscard]] inline std::complex<double> unit_root(std::size_t k,
                                                        std::size_t n)
    {
      const auto [re, im] {unit_root_parts(k, n)};
      return {unsigned_zero(re.hi), unsigned_zero(im.hi)};
    }

    /*!
     * \brief The roots `e^{−2πi k/n}` of one `n`, each as \ref unit_root rounds it, from two tables of about √n angles
     *        instead of a series a root: the angle's share within its octant (\ref reduce_to_octant) split as `a + b·m`,
     *        `m` the first table's length, and its cosine and sine those of the sum of the two tables' angles in
     *        double-double arithmetic, then placed by the octant and rounded once. The share 0 is the product of two
     *        exact `(1, 0)`, so the exact roots stay exact, and a root and its mirror image share their share, so they
     *        stay each other's conjugates to the bit.
     */
    class unit_roots
    {
    public:

      //! \brief The tables for the roots of \p n.
      explicit unit_roots(std::size_t n)
        : n_ {n}, m_ {static_cast<std::size_t>(std::sqrt(static_cast<double>(n))) + 1}
      {
        low_.reserve(m_);
        for (std::size_t a {0}; a < m_; ++a) {
          low_.push_back(octant_sin_cos(a, n));
        }
        for (std::size_t b {0}; b * m_ <= n; ++b) {
          high_.push_back(octant_sin_cos(b * m_, n));
        }
      }

      //! \brief `e^{−2πi k/n}` as double-double parts: `cos(A + B) = cA·cB − sA·sB`, `sin(A + B) = sA·cB + cA·sB`.
      [[nodiscard]] std::pair<double_double, double_double> parts(std::size_t k) const noexcept
      {
        const octant_share                             reduced                            {reduce_to_octant(k, n_)};
        const std::pair<double_double, double_double>& low                                {low_[reduced.share % m_]};
        const std::pair<double_double, double_double>& high                               {high_[reduced.share / m_]};
        const double_double                            cosine                             {dd_add(dd_mul(low.first, high.first), dd_negate(dd_mul(low.second, high.second)))};
        const double_double                            sine                               {dd_add(dd_mul(low.second, high.first), dd_mul(low.first, high.second))};
        return place_by_octant(reduced.octant, cosine, sine);
      }

      //! \brief \ref parts rounded once to `double`, a zero's sign positive.
      [[nodiscard]] std::complex<double> root(std::size_t k) const noexcept
      {
        const auto [re, im] {parts(k)};
        return {unsigned_zero(re.hi), unsigned_zero(im.hi)};
      }

    private:

      std::size_t                                          n_;    //!< The order of the roots.
      std::size_t                                          m_;    //!< The first table's length.
      std::vector<std::pair<double_double, double_double>> low_;  //!< cos and sin of (π/4)·a/n, a below m.
      std::vector<std::pair<double_double, double_double>> high_; //!< cos and sin of (π/4)·b·m/n, b·m up to n.
    };

    //! \brief `p · q` of complex numbers, each product rounded apart from the sum it enters (statements of their own).
    template <std::floating_point R>
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> rounded_product(const std::complex<R>& p,
                                                                          const std::complex<R>& q) noexcept
    {
      const R rr {p.real() * q.real()};
      const R ii {p.imag() * q.imag()};
      const R ri {p.real() * q.imag()};
      const R ir {p.imag() * q.real()};
      return {rr - ii, ri + ir};
    }

    //! \brief `w · v` of complex numbers on the vector paths: `w.re·v.re − w.im·v.im` and `w.re·v.im + w.im·v.re`, each
    //!        the second product rounded, the first fused with the sum by `std::fma`.
    template <std::floating_point R>
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> fused_product(const std::complex<R>& w,
                                                                        const std::complex<R>& v) noexcept
    {
      const R ii {w.imag() * v.imag()};
      const R ir {w.imag() * v.real()};
      return {std::fma(w.real(), v.real(), -ii), std::fma(w.real(), v.imag(), ir)};
    }

    //! \brief A butterfly's factor times \p v on the vector paths: \ref fused_product.
    template <bool Fused,
              std::floating_point R>
    requires Fused
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> twiddled(const std::complex<R>& w,
                                                                   const std::complex<R>& v) noexcept
    {
      return fused_product(w, v);
    }

    //! \brief A butterfly's factor times \p v on the portable path: \ref rounded_product.
    template <bool Fused,
              std::floating_point R>
    requires (!Fused)
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> twiddled(const std::complex<R>& w,
                                                                   const std::complex<R>& v) noexcept
    {
      return rounded_product(w, v);
    }

    //! \brief The largest prime a Stockham pass takes as its radix, by its own `r`-point sum: a length with a larger
    //!        prime factor is transformed by Bluestein's convolution.
    inline constexpr std::size_t fft_largest_radix {61};

    /*!
     * \brief The radix of each Stockham pass of \p n, or none where \p n has a prime factor past \ref fft_largest_radix:
     *        its powers of two first (4 first, its stride 1, then 16 while 16 points remain, then 4, then 2), then its
     *        factors 3, 5 and 7, then each other prime in ascending order.
     */
    [[nodiscard]] inline std::vector<std::size_t> fft_radices(std::size_t n)
    {
      std::vector<std::size_t> radices;
      std::size_t              twos {std::size_t {1} << std::countr_zero(n)};
      std::size_t              rest {n / twos};
      if (twos >= 4) {
        radices.push_back(4);
        twos /= 4;
      }
      for (; twos >= 16; twos /= 16) {
        radices.push_back(16);
      }
      for (; twos >= 4; twos /= 4) {
        radices.push_back(4);
      }
      if (twos == 2) {
        radices.push_back(2);
      }
      for (std::size_t prime {3}; rest > 1; prime += 2) {
        if (prime > fft_largest_radix) {
          return {}; // Bluestein's
        }
        for (; rest % prime == 0; rest /= prime) {
          radices.push_back(prime);
        }
      }
      return radices;
    }

    template <std::floating_point R>
    struct fft_plan;

    //! \brief Bluestein's convolution for a length \p n with a prime factor past \ref fft_largest_radix.
    template <std::floating_point R>
    struct bluestein_plan
    {
      std::size_t                        padded;   //!< M: the power of two at least 2n − 1 the convolution runs at.
      std::vector<std::complex<R>>       chirp;    //!< `e^{−πi j²/n}`, its exponent `j² mod 2n` reduced exactly.
      std::vector<std::complex<R>>       spectrum; //!< The transform of M points of the chirp's conjugate, wrapped.
      std::shared_ptr<const fft_plan<R>> inner;    //!< The plan of M points.
    };

    //! \brief The factors and passes of a transform of one length, built once and shared.
    template <std::floating_point R>
    struct fft_plan
    {
      std::size_t                       length;     //!< The transform's length.
      std::vector<std::size_t>          radices;    //!< Each Stockham pass' radix (\ref fft_radices).
      std::vector<std::complex<R>>      twiddles;
      //!< Each pass' factors from `p` = 1: its `w^p` run, then its `w^{2p}` run, to its `w^{(r−1)p}` run.
      std::vector<std::complex<R>>      radix_roots;
      //!< Each pass of a radix other than 4 and 16: `e^{−2πi m/r}` for m below r, the passes in order.
      std::array<std::complex<R>, 16>   sixteenths; //!< `e^{−2πi k/16}`: a radix-16 butterfly's own factors.
      std::optional<bluestein_plan<R>>  bluestein;  //!< Where the length has a prime factor past fft_largest_radix.
      std::size_t                       work;       //!< The working space a transform takes beside its data.
    };

    template <std::floating_point R>
    void forward_fft(const fft_plan<R>& plan,
                     std::complex<R>*   data,
                     std::complex<R>*   work,
                     bool               fused) noexcept;

    //! \brief The \ref fft_plan of length \p n.
    template <std::floating_point R>
    [[nodiscard]] fft_plan<R> make_fft_plan(std::size_t n)
    {
      using C = std::complex<R>;
      const unit_roots roots {n};
      const auto       narrow {[](std::complex<double> value) { return C {static_cast<R>(value.real()), static_cast<R>(value.imag())}; }};
      fft_plan<R>      plan {.length = n, .radices = fft_radices(n), .twiddles = {}, .radix_roots = {}, .sixteenths = {}, .bluestein = {}, .work = n};
      for (std::size_t k {0}; k < 16; ++k) {
        plan.sixteenths[k] = narrow(unit_root(k, 16));
      }
      if (plan.radices.empty() && n > 1) { // Bluestein's: X_k = c_k · Σ_j (x_j c_j) · conj(c_{k−j}), c_j = e^{−πi j²/n}
        bluestein_plan<R> b       {.padded = std::bit_ceil((2 * n) - 1), .chirp = {}, .spectrum = {}, .inner = {}};
        const unit_roots  doubled {2 * n};
        b.chirp.resize(n);
        for (std::size_t j {0}; j < n; ++j) {
          b.chirp[j] = narrow(doubled.root((j * j) % (2 * n)));
        }
        b.inner = std::make_shared<const fft_plan<R>>(make_fft_plan<R>(b.padded));
        b.spectrum.assign(b.padded, C {});
        for (std::size_t j {0}; j < n; ++j) {
          b.spectrum[j]                               = std::conj(b.chirp[j]);
          b.spectrum[(b.padded - j) % b.padded]       = std::conj(b.chirp[j]);
        }
        std::vector<C> scratch(b.padded);
        forward_fft(*b.inner, b.spectrum.data(), scratch.data(), false); // the portable arithmetic: the same on every path
        plan.work      = b.padded + b.inner->work;                       // the convolution's sequence, then its transforms' space
        plan.bluestein = std::move(b);
        return plan;
      }
      std::size_t current {n};
      std::size_t stride  {1};
      for (const std::size_t radix : plan.radices) {
        for (std::size_t power {1}; power < radix; ++power) {
          for (std::size_t p {1}; p < current / radix; ++p) { // p = 0 multiplies by nothing
            plan.twiddles.push_back(narrow(roots.root(power * p * stride)));
          }
        }
        if (radix != 4 && radix != 16) {
          for (std::size_t m {0}; m < radix; ++m) {
            plan.radix_roots.push_back(narrow(unit_root(m, radix)));
          }
        }
        current /= radix;
        stride  *= radix;
      }
      return plan;
    }

    //! \brief The plans kept, the least recently used dropped past it: a plan in use lives on in its holders.
    inline constexpr std::size_t fft_plans_kept {32};

    //! \brief The shared \ref fft_plan of length \p n, built on first use and kept among the \ref fft_plans_kept most
    //!        recently used.
    template <std::floating_point R>
    [[nodiscard]] std::shared_ptr<const fft_plan<R>> shared_fft_plan(std::size_t n)
    {
      struct entry
      {
        std::size_t                        length; //!< The plan's length.
        std::shared_ptr<const fft_plan<R>> plan;   //!< The plan.
      };
      static std::mutex         guard;
      static std::vector<entry> plans; // the most recently used last
      const std::scoped_lock    lock  {guard};
      const auto                found {std::find_if(plans.begin(), plans.end(), [n](const entry& e) { return e.length == n; })};
      if (found != plans.end()) {
        std::rotate(found, found + 1, plans.end());
        return plans.back().plan;
      }
      if (plans.size() >= fft_plans_kept) {
        plans.erase(plans.begin());
      }
      plans.push_back({n, std::make_shared<const fft_plan<R>>(make_fft_plan<R>(n))});
      return plans.back().plan;
    }

    /*!
     * \brief One Stockham pass of radix 4 over \p in into \p out: \p current points at stride \p stride, so
     *        `current · stride` is the length. Element `q + stride · (p + j · current/4)` of the input is the `j`-th
     *        input of butterfly `(p, q)`, whose outputs go to `q + stride · (4p + j)`, the last three times `w^{jp}`
     *        (\ref twiddled), none for `p` = 0.
     */
    template <bool Fused,
              std::floating_point R>
    void radix_four_pass(std::size_t            current,
                         std::size_t            stride,
                         const std::complex<R>* in,
                         std::complex<R>*       out,
                         const std::complex<R>* twiddles) noexcept
    {
      using C = std::complex<R>;
      const std::size_t quarter {current / 4};
      const std::size_t run     {quarter - 1}; // the factors of each power
      const std::size_t apart   {stride * quarter};
      const auto        butterfly {[&](std::size_t p, auto&& place) {
                                     const C* const in_p {in + (stride * p)};
                                     C* const       to   {out + (stride * 4 * p)};
                                     for (std::size_t q {0}; q < stride; ++q) {
                                       const C a           {in_p[q]};
                                       const C b           {in_p[q + apart]};
                                       const C c           {in_p[q + (2 * apart)]};
                                       const C d           {in_p[q + (3 * apart)]};
                                       const C a_plus_c    {a + c};
                                       const C a_minus_c   {a - c};
                                       const C b_plus_d    {b + d};
                                       const C b_minus_d   {b - d};
                                       const C i_b_minus_d {-b_minus_d.imag(), b_minus_d.real()}; // i·(b − d), exact
                                       to[q] = a_plus_c + b_plus_d;
                                       place(to + q, a_minus_c - i_b_minus_d, a_plus_c - b_plus_d, a_minus_c + i_b_minus_d);
                                     }
                                   }};
      butterfly(0, [stride](C* to, const C& first, const C& second, const C& third) {
                  to[stride]       = first;
                  to[2 * stride]   = second;
                  to[3 * stride]   = third;
                });
      for (std::size_t p {1}; p < quarter; ++p) {
        const C w1 {twiddles[p - 1]};
        const C w2 {twiddles[run + p - 1]};
        const C w3 {twiddles[(2 * run) + p - 1]};
        butterfly(p, [stride, w1, w2, w3](C* to, const C& first, const C& second, const C& third) {
                    to[stride]     = twiddled<Fused>(w1, first);
                    to[2 * stride] = twiddled<Fused>(w2, second);
                    to[3 * stride] = twiddled<Fused>(w3, third);
                  });
      }
    }

    //! \brief A 4-point transform of \p v0 to \p v3 into \p y: the radix-4 butterfly's sums, in its order.
    template <std::floating_point R>
    inline void four_point(const std::complex<R>& v0,
                           const std::complex<R>& v1,
                           const std::complex<R>& v2,
                           const std::complex<R>& v3,
                           std::complex<R>*       y) noexcept
    {
      using C = std::complex<R>;
      const C a_plus_c    {v0 + v2};
      const C a_minus_c   {v0 - v2};
      const C b_plus_d    {v1 + v3};
      const C b_minus_d   {v1 - v3};
      const C i_b_minus_d {-b_minus_d.imag(), b_minus_d.real()};
      y[0] = a_plus_c + b_plus_d;
      y[1] = a_minus_c - i_b_minus_d;
      y[2] = a_plus_c - b_plus_d;
      y[3] = a_minus_c + i_b_minus_d;
    }

    /*!
     * \brief One Stockham pass of radix 16 over \p in into \p out, as \ref radix_four_pass is of radix 4: each butterfly
     *        a 16-point transform as two of 4 points, `x_{j1 + 4·j2}` first transformed over `j2`, each result `(j1, k2)`
     *        then times `e^{−2πi j1·k2/16}` (\p sixteenths; `−i` exactly where `j1·k2` is 4), then over `j1` into output
     *        `4·k1 + k2`; output `k` of butterfly `p` times `w^{kp}`, none for `p` = 0. \p Fused as \ref twiddled.
     */
    template <bool Fused,
              std::floating_point R>
    void radix_sixteen_pass(std::size_t                            current,
                            std::size_t                            stride,
                            const std::complex<R>*                 in,
                            std::complex<R>*                       out,
                            const std::complex<R>*                 twiddles,
                            const std::array<std::complex<R>, 16>& sixteenths) noexcept
    {
      using C = std::complex<R>;
      const std::size_t part  {current / 16};
      const std::size_t run   {part - 1};
      const std::size_t apart {stride * part};
      for (std::size_t p {0}; p < part; ++p) {
        const C* const in_p {in + (stride * p)};
        C* const       to   {out + (stride * 16 * p)};
        for (std::size_t q {0}; q < stride; ++q) {
          std::array<C, 16> inner {};
          #  pragma GCC unroll 16
          for (std::size_t j1 {0}; j1 < 4; ++j1) {
            four_point(in_p[q + (j1 * apart)], in_p[q + ((j1 + 4) * apart)], in_p[q + ((j1 + 8) * apart)], in_p[q + ((j1 + 12) * apart)],
                       inner.data() + (4 * j1));
            #  pragma GCC unroll 16
            for (std::size_t k2 {1}; k2 < 4 && j1 != 0; ++k2) {
              C& value {inner[(4 * j1) + k2]};
              value = (j1 * k2) == 4 ? C {value.imag(), -value.real()} : twiddled<Fused>(sixteenths[j1 * k2], value);
            }
          }
          #  pragma GCC unroll 16
          for (std::size_t k2 {0}; k2 < 4; ++k2) {
            std::array<C, 4> outputs {};
            four_point(inner[k2], inner[4 + k2], inner[8 + k2], inner[12 + k2], outputs.data());
            #  pragma GCC unroll 16
            for (std::size_t k1 {0}; k1 < 4; ++k1) {
              const std::size_t k {(4 * k1) + k2};
              to[q + (stride * k)] = (p == 0 || k == 0) ? outputs[k1] : twiddled<Fused>(twiddles[((k - 1) * run) + p - 1], outputs[k1]);
            }
          }
        }
      }
    }

    /*!
     * \brief One Stockham pass of any radix \p radix, as \ref radix_four_pass is of 4: butterfly `(p, q)` the `r`-point
     *        transform of its inputs, `X_k = x_0 + Σ_{j ≥ 1} e^{−2πi jk/r} · x_j` in ascending `j` (\p roots; a factor
     *        of 1 multiplies nothing, one of −1 subtracts), output `k` then times `w^{kp}`, none for `p` = 0 or `k` = 0.
     *        Inlined into each caller, so that the x86-64 fused one is built for FMA.
     */
    template <bool Fused,
              std::floating_point R>
    NUMERICAL_ALWAYS_INLINE void any_radix_pass_body(std::size_t            radix,
                                                     std::size_t            current,
                                                     std::size_t            stride,
                                                     const std::complex<R>* in,
                                                     std::complex<R>*       out,
                                                     const std::complex<R>* twiddles,
                                                     const std::complex<R>* roots) noexcept
    {
      using C = std::complex<R>;
      const std::size_t part  {current / radix};
      const std::size_t run   {part - 1};
      const std::size_t apart {stride * part};
      for (std::size_t p {0}; p < part; ++p) {
        const C* const in_p {in + (stride * p)};
        C* const       to   {out + (stride * radix * p)};
        for (std::size_t q {0}; q < stride; ++q) {
          for (std::size_t k {0}; k < radix; ++k) {
            C           sum   {in_p[q]};
            std::size_t power {0}; // j·k mod r, kept by additions
            for (std::size_t j {1}; j < radix; ++j) {
              power += k;
              power -= power >= radix ? radix : 0;
              const C value {in_p[q + (j * apart)]};
              if (power == 0) {
                sum += value;
              }
              else if (2 * power == radix) { // e^{−πi} = −1: a difference, exact
                sum -= value;
              }
              else {
                sum += twiddled<Fused>(roots[power], value);
              }
            }
            to[q + (stride * k)] = (p == 0 || k == 0) ? sum : twiddled<Fused>(twiddles[((k - 1) * run) + p - 1], sum);
          }
        }
      }
    }

    //! \brief \ref any_radix_pass_body, its products as \p Fused.
    template <bool Fused,
              std::floating_point R>
    void any_radix_pass(std::size_t            radix,
                        std::size_t            current,
                        std::size_t            stride,
                        const std::complex<R>* in,
                        std::complex<R>*       out,
                        const std::complex<R>* twiddles,
                        const std::complex<R>* roots) noexcept
    {
      any_radix_pass_body<Fused>(radix, current, stride, in, out, twiddles, roots);
    }

#ifdef NUMERICAL_X86_COMPILED
    //! \brief \ref any_radix_pass_body fused, built for AVX2 with FMA: each `std::fma` one instruction.
    template <std::floating_point R>
    NUMERICAL_TARGET("avx2,fma") void any_radix_pass_x86(std::size_t            radix,
                                                         std::size_t            current,
                                                         std::size_t            stride,
                                                         const std::complex<R>* in,
                                                         std::complex<R>*       out,
                                                         const std::complex<R>* twiddles,
                                                         const std::complex<R>* roots) noexcept
    {
      any_radix_pass_body<true>(radix, current, stride, in, out, twiddles, roots);
    }

#endif

    //! \brief The last pass of an odd power of two: butterflies of radix 2 at stride \p stride, no factor.
    template <std::floating_point R>
    void radix_two_pass(std::size_t            stride,
                        const std::complex<R>* in,
                        std::complex<R>*       out) noexcept
    {
      for (std::size_t q {0}; q < stride; ++q) {
        const std::complex<R> a {in[q]};
        const std::complex<R> b {in[q + stride]};
        out[q]          = a + b;
        out[q + stride] = a - b;
      }
    }

    /*!
     * \brief The Stockham passes of the \p plan over \p data, \p work as long, the result in \p data: radix 4 by \p four
     *        (\ref radix_four_pass's arguments), 16 by \p sixteen (\ref radix_sixteen_pass's), 2 by \ref radix_two_pass
     *        where it is the last pass, any other by \p other (\ref any_radix_pass's).
     */
    template <std::floating_point R,
              typename Four,
              typename Sixteen,
              typename Other>
    void stockham_passes(const fft_plan<R>& plan,
                         std::complex<R>*   data,
                         std::complex<R>*   work,
                         Four&&             four,
                         Sixteen&&          sixteen,
                         Other&&            other) noexcept
    {
      const std::size_t      n        {plan.length};
      std::complex<R>*       in       {data};
      std::complex<R>*       out      {work};
      const std::complex<R>* twiddles {plan.twiddles.data()};
      const std::complex<R>* roots    {plan.radix_roots.data()};
      std::size_t            current  {n};
      std::size_t            stride   {1};
      for (const std::size_t radix : plan.radices) {
        if (radix == 16) {
          sixteen(current, stride, in, out, twiddles, plan.sixteenths);
        }
        else if (radix == 4) {
          four(current, stride, in, out, twiddles);
        }
        else if (radix == 2 && current == 2) { // the last pass: no factor
          radix_two_pass(stride, in, out);
          roots += radix;
        }
        else {
          other(radix, current, stride, in, out, twiddles, roots);
          roots += radix;
        }
        twiddles += (radix - 1) * (current / radix - 1);
        current  /= radix;
        stride   *= radix;
        std::swap(in, out);
      }
      if (in != data) {
        std::copy(in, in + n, data);
      }
    }

    //! \brief Whether a transform's butterflies fuse their products (\ref fused_product): on every path past the portable.
    [[nodiscard]] inline bool fft_fuses() noexcept
    {
      return gemm_path_current() != gemm_path::portable;
    }

    /*!
     * \brief Bluestein's transform of the \p plan's length in place in \p data, \p work its \ref fft_plan::work: the
     *        input times the chirp, padded to M; its transform times the chirp's spectrum; the inverse transform of that
     *        (`conj(fft(conj ·))`, then 1/M, exact), times the chirp. Products as \p Fused. Inlined into each caller.
     */
    template <bool Fused,
              std::floating_point R>
    NUMERICAL_ALWAYS_INLINE void bluestein_body(const fft_plan<R>& plan,
                                                std::complex<R>*   data,
                                                std::complex<R>*   work) noexcept
    {
      using C = std::complex<R>;
      const bluestein_plan<R>& b     {*plan.bluestein};
      const std::size_t        n     {plan.length};
      const std::size_t        m     {b.padded};
      const R                  scale {R(1) / static_cast<R>(m)};
      C* const                 a     {work};
      for (std::size_t j {0}; j < n; ++j) {
        a[j] = twiddled<Fused>(b.chirp[j], data[j]);
      }
      std::fill(a + n, a + m, C {});
      forward_fft(*b.inner, a, work + m, Fused);
      for (std::size_t k {0}; k < m; ++k) {
        a[k] = std::conj(twiddled<Fused>(b.spectrum[k], a[k]));
      }
      forward_fft(*b.inner, a, work + m, Fused);
      for (std::size_t k {0}; k < n; ++k) {
        const C value {a[k].real() * scale, -a[k].imag() * scale};
        data[k] = twiddled<Fused>(b.chirp[k], value);
      }
    }

#ifdef NUMERICAL_X86_COMPILED
    //! \brief \ref bluestein_body fused, built for AVX2 with FMA.
    template <std::floating_point R>
    NUMERICAL_TARGET("avx2,fma") void bluestein_x86(const fft_plan<R>& plan,
                                                    std::complex<R>*   data,
                                                    std::complex<R>*   work) noexcept
    {
      bluestein_body<true>(plan, data, work);
    }

#endif

    /*!
     * \brief The forward transform of the \p plan's length, in place in \p data, \p work its \ref fft_plan::work: its
     *        Stockham passes, or Bluestein's convolution, the products fused where \p fused (\ref fft_fuses).
     */
    template <std::floating_point R>
    void forward_fft(const fft_plan<R>& plan,
                     std::complex<R>*   data,
                     std::complex<R>*   work,
                     bool               fused) noexcept
    {
      if (plan.bluestein) {
        if (!fused) {
          bluestein_body<false>(plan, data, work);
          return;
        }
#ifdef NUMERICAL_X86_COMPILED
        bluestein_x86(plan, data, work);
#else
        bluestein_body<true>(plan, data, work);
#endif
        return;
      }
      if (!fused) {
        stockham_passes(plan, data, work, radix_four_pass<false, R>, radix_sixteen_pass<false, R>, any_radix_pass<false, R>);
        return;
      }
#ifdef NUMERICAL_X86_COMPILED
      stockham_passes(plan, data, work, avx2_radix_four_pass<R>, avx2_radix_sixteen_pass<R>, any_radix_pass_x86<R>);
#else
      stockham_passes(plan, data, work, radix_four_pass<true, R>, radix_sixteen_pass<true, R>, any_radix_pass<true, R>);
#endif
    }

    //! \brief The complex type a transform of \p T computes in: \p T itself, or a real type's complex.
    template <typename T>
    struct fft_complex
    {
      using type = std::complex<T>; //!< A real `float` or `double`'s.
    };

    //! \brief \ref fft_complex of a complex type: itself.
    template <split_complex C>
    struct fft_complex<C>
    {
      using type = C; //!< The type itself.
    };

    //! \brief The types \ref numerical::fft transforms: `float`, `double` and their complex numbers.
    template <typename T>
    concept fft_input = std::is_same_v<T, float> || std::is_same_v<T, double> || split_complex<T>;

    //! \brief The scale of a transform's output for \p norm, forward or not, of length \p n: 1 where none.
    template <std::floating_point R>
    [[nodiscard]] R fft_scale(fft_norm    norm,
                              bool        inverse,
                              std::size_t n)
    {
      if (norm == fft_norm::ortho) {
        return R(1) / std::sqrt(static_cast<R>(n));
      }
      return (norm == fft_norm::forward) != inverse ? R(1) / static_cast<R>(n) : R(1);
    }

    //! \brief The most elements \ref fft_scratch keeps on a thread between transforms: 1 MiB of complex `double`.
    inline constexpr std::size_t fft_scratch_kept {std::size_t {1} << 16};

    //! \brief A transform's working space of \p C: the calling thread's own, kept for the next transform, up to
    //!        \ref fft_scratch_kept elements; past it, its own allocation.
    template <split_complex C>
    class fft_scratch
    {
    public:

      //! \brief Space for \p count elements.
      explicit fft_scratch(std::size_t count)
      {
        if (count > fft_scratch_kept) {
          own_.resize(count);
          data_ = own_.data();
          return;
        }
        thread_local buffer<C> kept;
        if (kept.size() < count) {
          kept.resize(count); // the most this thread has needed, up to fft_scratch_kept
        }
        data_ = kept.data();
      }

      //! \brief The first element.
      [[nodiscard]] C* data() const noexcept
      {
        return data_;
      }

    private:

      buffer<C> own_;            //!< The space past fft_scratch_kept.
      C*        data_ {nullptr}; //!< The space in use.
    };

    /*!
     * \brief The transform of \p a along \p axis, \p n points (the axis cut or padded with zeros to it, as NumPy's),
     *        forward or \p inverse (`conj(fft(conj x))`, exact), scaled for \p norm. Each line gathered, transformed
     *        and scattered on its own, the lines shared among the threads: a line's bits are its own.
     */
    template <fft_input T>
    [[nodiscard]] tensor<typename fft_complex<T>::type> transform_axis(const tensor<T>&           a,
                                                                       std::optional<std::size_t> n,
                                                                       std::ptrdiff_t             axis,
                                                                       fft_norm                   norm,
                                                                       bool                       inverse)
    {
      using C = typename fft_complex<T>::type;
      using R = typename C::value_type;
      const auto rank {static_cast<std::ptrdiff_t>(a.rank())};
      if (axis < -rank || axis >= rank) {
        throw dimension_error("fft: axis " + std::to_string(axis) + " is out of bounds for a tensor of rank " + std::to_string(rank));
      }
      const auto        index  {static_cast<std::size_t>(axis < 0 ? axis + rank : axis)};
      const std::size_t extent {a.shape()[index]};
      const std::size_t points {n.value_or(extent)};
      if (points == 0) {
        throw dimension_error("fft: invalid number of data points (0)");
      }
      std::vector<std::size_t> shape {a.shape()};
      shape[index] = points;
      const axis_walk walk           {walk_along(shape, index)};
      const R         scale          {fft_scale<R>(norm, inverse, points)};
      buffer<C>       out(element_count(shape.begin(), shape.end()));
      if (walk.outer == 0) {
        return tensor<C> {std::move(shape), std::move(out)};
      }
      const std::shared_ptr<const fft_plan<R>> plan  {shared_fft_plan<R>(points)};
      const std::size_t                        kept  {std::min(points, extent)};
      const bool                               fused {fft_fuses()};
      const T* const                           from  {a.data().data()};
      parallel_for(walk.outer * walk.inner, 5 * points * static_cast<std::size_t>(std::bit_width(points)), [&](std::size_t line) {
                     const std::size_t outer {line / walk.inner};
                     const std::size_t inner {line % walk.inner};
                     const T* const    in    {from + (outer * extent * walk.inner) + inner};
                     C* const          to    {out.data() + (outer * points * walk.inner) + inner};
                     fft_scratch<C>    scratch {plan->work + (walk.inner == 1 ? 0 : points)};
                     C* const          data  {walk.inner == 1 ? to : scratch.data() + plan->work}; // a line of the output where it lies together
                     for (std::size_t k {0}; k < points; ++k) {
                       const C value {k < kept ? C(in[k * walk.inner]) : C {}};
                       data[k] = inverse ? std::conj(value) : value;
                     }
                     forward_fft(*plan, data, scratch.data(), fused);
                     if (data == to && !inverse && scale == R(1)) {
                       return; // transformed where it lies, nothing to conjugate or scale
                     }
                     for (std::size_t k {0}; k < points; ++k) {
                       const C value {inverse ? std::conj(data[k]) : data[k]};
                       to[k * walk.inner] = scale == R(1) ? value : C {value.real() * scale, value.imag() * scale};
                     }
                   });
      return tensor<C> {std::move(shape), std::move(out)};
    }
  } // namespace detail

  /*!
   * \brief The discrete Fourier transform of \p a along \p axis, as `numpy.fft.fft`: `X_k = Σ_j x_j e^{−2πi jk/n}`.
   * \param[in] a    A tensor of `float`, `double` or their complex numbers; a real one is transformed as complex.
   * \param[in] n    The points transformed: the axis cut to them, or padded with zeros; its extent by default.
   * \param[in] axis The axis transformed, negative from the last (the last by default).
   * \param[in] norm The scaling (\ref fft_norm): none by default.
   * \return The complex transform, of `a`'s shape with the axis `n` long.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <detail::fft_input T>
  [[nodiscard]] tensor<typename detail::fft_complex<T>::type> fft(const tensor<T>&           a,
                                                                  std::optional<std::size_t> n    = std::nullopt,
                                                                  std::ptrdiff_t             axis = -1,
                                                                  fft_norm                   norm = fft_norm::backward)
  {
    return detail::transform_axis(a, n, axis, norm, false);
  }

  /*!
   * \brief The inverse discrete Fourier transform of \p a along \p axis, as `numpy.fft.ifft`:
   *        `x_j = (1/n) Σ_k X_k e^{2πi jk/n}` with the default \p norm.
   * \param[in] a    A tensor of `float`, `double` or their complex numbers.
   * \param[in] n    The points transformed: the axis cut to them, or padded with zeros; its extent by default.
   * \param[in] axis The axis transformed, negative from the last (the last by default).
   * \param[in] norm The scaling (\ref fft_norm): 1/n by default.
   * \return The complex inverse transform, of `a`'s shape with the axis `n` long.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <detail::fft_input T>
  [[nodiscard]] tensor<typename detail::fft_complex<T>::type> ifft(const tensor<T>&           a,
                                                                   std::optional<std::size_t> n    = std::nullopt,
                                                                   std::ptrdiff_t             axis = -1,
                                                                   fft_norm                   norm = fft_norm::backward)
  {
    return detail::transform_axis(a, n, axis, norm, true);
  }
} // namespace numerical

#endif // NUMERICAL_FFT_HPP
