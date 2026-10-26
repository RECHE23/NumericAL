/*!
 * \file fft.hpp
 * \brief The discrete Fourier transform of a tensor along an axis, as `numpy.fft.fft` and `numpy.fft.ifft` compute it.
 *
 * Every length runs in O(n log n). A length whose prime factors are at most 61 runs Stockham passes, each reading and
 * writing in order, their radices fixed by the length alone (`fft_radices`): its powers of two first (4, then 16 while
 * 16 points remain, then 4), then 3, 5 and 7, then each other prime, a lone factor 2 last. Radices 3 and 5 take their own
 * butterflies, a larger prime its r-point sum. A length with a larger prime factor runs Bluestein's convolution over
 * three transforms of a power of two. The twiddle factors are computed in double-double arithmetic by one fixed sequence
 * of operations, each a sum or an explicit fused multiply-add that no compiler may contract otherwise, then rounded once:
 * the same factors on every machine, whatever its `sin` and `cos`.
 *
 * Two sets of bits, each the same on every machine and compiler, any number of threads, a line alone or among others,
 * whatever its axis: the portable path's, whose butterflies round each product apart from its sum (built, as the
 * library's builds are, with no contraction across statements: `-ffp-contract=on` or `off`), and the vector paths'
 * (NEON, AMX, AVX2, AVX-512 alike), which fuse them. A change to a length's radices or to a butterfly's arithmetic changes
 * that length's last bits, and is noted as such.
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
#include "fft_kernels.hpp"
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

    //! \brief \p x, held in a register as its rounded value: GCC on arm64 with the complex-arithmetic unit (ARMv8.3's
    //!        FCMLA: `-mcpu=apple-m1`, Neoverse V1) recognizes `a·b − c·d, a·d + c·b` as a complex product and fuses one
    //!        product into the other's sum whatever `-ffp-contract` says; elsewhere \p x as it is.
    template <std::floating_point R>
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE R rounded_alone(R x) noexcept
    {
#if defined(__GNUC__) && !defined(__clang__) && defined(__aarch64__)
      __asm__ ("" : "+w" (x)); // NOLINT(hicpp-no-assembler): no builtin keeps a value from GCC's complex patterns
#endif
      return x;
    }

    //! \brief `p · q` of complex numbers, each product rounded apart from the sum it enters (statements of their own,
    //!        \ref rounded_alone).
    template <std::floating_point R>
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> rounded_product(const std::complex<R>& p,
                                                                          const std::complex<R>& q) noexcept
    {
      const R rr {p.real() * q.real()};
      const R ii {rounded_alone(p.imag() * q.imag())};
      const R ri {p.real() * q.imag()};
      const R ir {rounded_alone(p.imag() * q.real())};
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

    // The lanes' functions: built for AVX2 with FMA on x86-64, and there inlined by the compiler's choice where the passes'
    // bodies, built for no unit of their own, are inlined into their callers built for those units (an always-inlined
    // function of a unit its caller lacks is refused); always inlined elsewhere.
#if defined(NUMERICAL_X86_COMPILED)
#  define NUMERICAL_FFT_LANES_TARGET NUMERICAL_TARGET("avx2,fma")          // NOLINT(cppcoreguidelines-macro-usage): an attribute, which no function can stand for
#  if defined(_MSC_VER) && !defined(__clang__)
#    define NUMERICAL_FFT_LANES_INLINE NUMERICAL_ALWAYS_INLINE             // NOLINT(cppcoreguidelines-macro-usage): the same
#  else
#    define NUMERICAL_FFT_LANES_INLINE NUMERICAL_TARGET("avx2,fma") inline // NOLINT(cppcoreguidelines-macro-usage): the same
#  endif
#else
#  define NUMERICAL_FFT_LANES_TARGET
#  define NUMERICAL_FFT_LANES_INLINE NUMERICAL_ALWAYS_INLINE // NOLINT(cppcoreguidelines-macro-usage): the same
#endif

#if defined(NUMERICAL_X86_COMPILED) || defined(NUMERICAL_NEON_COMPILED)
#  define NUMERICAL_FFT_LANES // lines transformed a register of them at a time (tested with #ifdef only)

#  if defined(NUMERICAL_X86_COMPILED)
    //! \brief A register of lines' parts on x86-64: four `double` (AVX2).
    using fft_lanes_double = __m256d;
    //! \brief A register of lines' parts on x86-64: eight `float` (AVX2).
    using fft_lanes_float = __m256;

    //! \brief `a + b`, lane by lane.
    NUMERICAL_FFT_LANES_INLINE __m256d lanes_add(__m256d a,
                                                 __m256d b) noexcept
    {
      return avx2_add(a, b);
    }

    //! \brief \ref lanes_add of `float`.
    NUMERICAL_FFT_LANES_INLINE __m256 lanes_add(__m256 a,
                                                __m256 b) noexcept
    {
      return avx2_add(a, b);
    }

    //! \brief `a − b`, lane by lane.
    NUMERICAL_FFT_LANES_INLINE __m256d lanes_sub(__m256d a,
                                                 __m256d b) noexcept
    {
      return avx2_sub(a, b);
    }

    //! \brief \ref lanes_sub of `float`.
    NUMERICAL_FFT_LANES_INLINE __m256 lanes_sub(__m256 a,
                                                __m256 b) noexcept
    {
      return avx2_sub(a, b);
    }

    //! \brief `a · b`, lane by lane, rounded.
    NUMERICAL_FFT_LANES_INLINE __m256d lanes_mul(__m256d a,
                                                 __m256d b) noexcept
    {
      return avx2_mul(a, b);
    }

    //! \brief \ref lanes_mul of `float`.
    NUMERICAL_FFT_LANES_INLINE __m256 lanes_mul(__m256 a,
                                                __m256 b) noexcept
    {
      return avx2_mul(a, b);
    }

    //! \brief `a · b + c`, lane by lane, one rounding.
    NUMERICAL_FFT_LANES_INLINE __m256d lanes_fused(__m256d a,
                                                   __m256d b,
                                                   __m256d c) noexcept
    {
      return avx2_fma(a, b, c);
    }

    //! \brief \ref lanes_fused of `float`.
    NUMERICAL_FFT_LANES_INLINE __m256 lanes_fused(__m256 a,
                                                  __m256 b,
                                                  __m256 c) noexcept
    {
      return avx2_fma(a, b, c);
    }

    //! \brief `−a`, lane by lane, exact.
    NUMERICAL_FFT_LANES_INLINE __m256d lanes_negated(__m256d a) noexcept
    {
      return avx2_negated(a);
    }

    //! \brief \ref lanes_negated of `float`.
    NUMERICAL_FFT_LANES_INLINE __m256 lanes_negated(__m256 a) noexcept
    {
      return avx2_negated(a);
    }

    //! \brief \p c in every lane.
    NUMERICAL_FFT_LANES_INLINE __m256d lanes_splat(double c) noexcept
    {
      return avx2_splat(c);
    }

    //! \brief \ref lanes_splat of `float`.
    NUMERICAL_FFT_LANES_INLINE __m256 lanes_splat(float c) noexcept
    {
      return avx2_splat(c);
    }

    //! \brief The complex `float` at \p from and \p apart past it, as four `float`.
    NUMERICAL_FFT_LANES_INLINE __m128 lanes_two(const double* from,
                                                std::size_t   apart) noexcept
    {
      return _mm_castpd_ps(_mm_loadh_pd(_mm_load_sd(from), from + apart));
    }

    //! \brief The parts of the complex numbers at \p at, \p apart from one another (four of them), into \p re and \p im.
    NUMERICAL_FFT_LANES_INLINE void lanes_read(const std::complex<double>* at,
                                               std::size_t                 apart,
                                               __m256d&                    re,
                                               __m256d&                    im) noexcept
    {
      const double* const from  {reinterpret_cast<const double*>(at)};                                                  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
      const __m256d       even  {_mm256_set_m128d(_mm_loadu_pd(from + (4 * apart)), _mm_loadu_pd(from))};               // r0 i0 | r2 i2
      const __m256d       odd   {_mm256_set_m128d(_mm_loadu_pd(from + (6 * apart)), _mm_loadu_pd(from + (2 * apart)))}; // r1 i1 | r3 i3
      re = _mm256_unpacklo_pd(even, odd);                                                                               // r0 r1 | r2 r3
      im = _mm256_unpackhi_pd(even, odd);
    }

    //! \brief \ref lanes_read of eight complex `float`.
    NUMERICAL_FFT_LANES_INLINE void lanes_read(const std::complex<float>* at,
                                               std::size_t                apart,
                                               __m256&                    re,
                                               __m256&                    im) noexcept
    {
      const double* const from {reinterpret_cast<const double*>(at)};                                                         // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex float is eight bytes, read whole
      const __m256        low  {_mm256_set_m128(lanes_two(from + (4 * apart), apart), lanes_two(from, apart))};               // r0 i0 r1 i1 | r4 i4 r5 i5
      const __m256        high {_mm256_set_m128(lanes_two(from + (6 * apart), apart), lanes_two(from + (2 * apart), apart))}; // r2 i2 r3 i3 | r6 i6 r7 i7
      re = _mm256_shuffle_ps(low, high, 0x88);                                                                                // r0 r1 r2 r3 | r4 r5 r6 r7
      im = _mm256_shuffle_ps(low, high, 0xDD);
    }

    //! \brief The complex numbers of parts \p re and \p im stored at \p at, \p apart from one another (four of them).
    NUMERICAL_FFT_LANES_INLINE void lanes_write(std::complex<double>* at,
                                                std::size_t           apart,
                                                __m256d               re,
                                                __m256d               im) noexcept
    {
      double* const into {reinterpret_cast<double*>(at)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
      const __m256d even {_mm256_unpacklo_pd(re, im)};    // r0 i0 | r2 i2
      const __m256d odd  {_mm256_unpackhi_pd(re, im)};    // r1 i1 | r3 i3
      _mm_storeu_pd(into, _mm256_castpd256_pd128(even));
      _mm_storeu_pd(into + (2 * apart), _mm256_castpd256_pd128(odd));
      _mm_storeu_pd(into + (4 * apart), _mm256_extractf128_pd(even, 1));
      _mm_storeu_pd(into + (6 * apart), _mm256_extractf128_pd(odd, 1));
    }

    //! \brief \ref lanes_write of eight complex `float`.
    NUMERICAL_FFT_LANES_INLINE void lanes_write(std::complex<float>* at,
                                                std::size_t          apart,
                                                __m256               re,
                                                __m256               im) noexcept
    {
      double* const into {reinterpret_cast<double*>(at)};                // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex float is eight bytes, written whole
      const __m256d low  {_mm256_castps_pd(_mm256_unpacklo_ps(re, im))}; // r0 i0 r1 i1 | r4 i4 r5 i5
      const __m256d high {_mm256_castps_pd(_mm256_unpackhi_ps(re, im))}; // r2 i2 r3 i3 | r6 i6 r7 i7
      const __m128d l0   {_mm256_castpd256_pd128(low)};
      const __m128d l1   {_mm256_extractf128_pd(low, 1)};
      const __m128d h0   {_mm256_castpd256_pd128(high)};
      const __m128d h1   {_mm256_extractf128_pd(high, 1)};
      _mm_storel_pd(into, l0);
      _mm_storeh_pd(into + apart, l0);
      _mm_storel_pd(into + (2 * apart), h0);
      _mm_storeh_pd(into + (3 * apart), h0);
      _mm_storel_pd(into + (4 * apart), l1);
      _mm_storeh_pd(into + (5 * apart), l1);
      _mm_storel_pd(into + (6 * apart), h1);
      _mm_storeh_pd(into + (7 * apart), h1);
    }

#  else
    //! \brief A register of lines' parts on arm64: two `double` (NEON).
    using fft_lanes_double = float64x2_t;
    //! \brief A register of lines' parts on arm64: four `float` (NEON).
    using fft_lanes_float = float32x4_t;

    //! \brief `a + b`, lane by lane.
    NUMERICAL_ALWAYS_INLINE float64x2_t lanes_add(float64x2_t a,
                                                  float64x2_t b) noexcept
    {
      return vaddq_f64(a, b);
    }

    //! \brief \ref lanes_add of `float`.
    NUMERICAL_ALWAYS_INLINE float32x4_t lanes_add(float32x4_t a,
                                                  float32x4_t b) noexcept
    {
      return vaddq_f32(a, b);
    }

    //! \brief `a − b`, lane by lane.
    NUMERICAL_ALWAYS_INLINE float64x2_t lanes_sub(float64x2_t a,
                                                  float64x2_t b) noexcept
    {
      return vsubq_f64(a, b);
    }

    //! \brief \ref lanes_sub of `float`.
    NUMERICAL_ALWAYS_INLINE float32x4_t lanes_sub(float32x4_t a,
                                                  float32x4_t b) noexcept
    {
      return vsubq_f32(a, b);
    }

    //! \brief `a · b`, lane by lane, rounded.
    NUMERICAL_ALWAYS_INLINE float64x2_t lanes_mul(float64x2_t a,
                                                  float64x2_t b) noexcept
    {
      return vmulq_f64(a, b);
    }

    //! \brief \ref lanes_mul of `float`.
    NUMERICAL_ALWAYS_INLINE float32x4_t lanes_mul(float32x4_t a,
                                                  float32x4_t b) noexcept
    {
      return vmulq_f32(a, b);
    }

    //! \brief `a · b + c`, lane by lane, one rounding.
    NUMERICAL_ALWAYS_INLINE float64x2_t lanes_fused(float64x2_t a,
                                                    float64x2_t b,
                                                    float64x2_t c) noexcept
    {
      return vfmaq_f64(c, a, b);
    }

    //! \brief \ref lanes_fused of `float`.
    NUMERICAL_ALWAYS_INLINE float32x4_t lanes_fused(float32x4_t a,
                                                    float32x4_t b,
                                                    float32x4_t c) noexcept
    {
      return vfmaq_f32(c, a, b);
    }

    //! \brief `−a`, lane by lane, exact.
    NUMERICAL_ALWAYS_INLINE float64x2_t lanes_negated(float64x2_t a) noexcept
    {
      return vnegq_f64(a);
    }

    //! \brief \ref lanes_negated of `float`.
    NUMERICAL_ALWAYS_INLINE float32x4_t lanes_negated(float32x4_t a) noexcept
    {
      return vnegq_f32(a);
    }

    //! \brief \p c in every lane.
    NUMERICAL_ALWAYS_INLINE float64x2_t lanes_splat(double c) noexcept
    {
      return vdupq_n_f64(c);
    }

    //! \brief \ref lanes_splat of `float`.
    NUMERICAL_ALWAYS_INLINE float32x4_t lanes_splat(float c) noexcept
    {
      return vdupq_n_f32(c);
    }

    //! \brief The parts of the complex numbers at \p at, \p apart from one another (two of them), into \p re and \p im.
    NUMERICAL_ALWAYS_INLINE void lanes_read(const std::complex<double>* at,
                                            std::size_t                 apart,
                                            float64x2_t&                re,
                                            float64x2_t&                im) noexcept
    {
      const double* const from   {reinterpret_cast<const double*>(at)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
      const float64x2_t   first  {vld1q_f64(from)};
      const float64x2_t   second {vld1q_f64(from + (2 * apart))};
      re = vzip1q_f64(first, second);
      im = vzip2q_f64(first, second);
    }

    //! \brief \ref lanes_read of four complex `float`.
    NUMERICAL_ALWAYS_INLINE void lanes_read(const std::complex<float>* at,
                                            std::size_t                apart,
                                            float32x4_t&               re,
                                            float32x4_t&               im) noexcept
    {
      const float* const from {reinterpret_cast<const float*>(at)};                                       // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
      const float32x4_t  low  {vcombine_f32(vld1_f32(from), vld1_f32(from + (2 * apart)))};               // r0 i0 r1 i1
      const float32x4_t  high {vcombine_f32(vld1_f32(from + (4 * apart)), vld1_f32(from + (6 * apart)))}; // r2 i2 r3 i3
      re = vuzp1q_f32(low, high);
      im = vuzp2q_f32(low, high);
    }

    //! \brief The complex numbers of parts \p re and \p im stored at \p at, \p apart from one another (two of them).
    NUMERICAL_ALWAYS_INLINE void lanes_write(std::complex<double>* at,
                                             std::size_t           apart,
                                             float64x2_t           re,
                                             float64x2_t           im) noexcept
    {
      double* const into {reinterpret_cast<double*>(at)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
      vst1q_f64(into, vzip1q_f64(re, im));
      vst1q_f64(into + (2 * apart), vzip2q_f64(re, im));
    }

    //! \brief \ref lanes_write of four complex `float`.
    NUMERICAL_ALWAYS_INLINE void lanes_write(std::complex<float>* at,
                                             std::size_t          apart,
                                             float32x4_t          re,
                                             float32x4_t          im) noexcept
    {
      float* const      into  {reinterpret_cast<float*>(at)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
      const float32x4_t first {vzip1q_f32(re, im)};           // r0 i0 r1 i1
      const float32x4_t last  {vzip2q_f32(re, im)};           // r2 i2 r3 i3
      vst1_f32(into, vget_low_f32(first));
      vst1_f32(into + (2 * apart), vget_high_f32(first));
      vst1_f32(into + (4 * apart), vget_low_f32(last));
      vst1_f32(into + (6 * apart), vget_high_f32(last));
    }

#  endif

    //! \brief The register of lines' parts of \p R: \ref fft_lanes_double or \ref fft_lanes_float.
    template <std::floating_point R>
    struct fft_lanes_register;

    //! \brief \ref fft_lanes_double.
    template <>
    struct fft_lanes_register<double>
    {
      using type = fft_lanes_double; //!< The register.
    };

    //! \brief \ref fft_lanes_float.
    template <>
    struct fft_lanes_register<float>
    {
      using type = fft_lanes_float; //!< The register.
    };

    //! \brief \ref fft_lanes_register's register.
    template <std::floating_point R>
    using fft_lanes_of = typename fft_lanes_register<R>::type;

    /*!
     * \brief One part of a register of lines' complex numbers, a line a lane (\ref fft_lanes), \p R the lanes' type:
     *        each operation each lane's own, rounded as the scalar operation rounds it.
     */
    template <std::floating_point R>
    struct alignas(sizeof(fft_lanes_of<R>)) fft_pack // a register's own alignment, which GCC drops from an alias of it
    {
      fft_lanes_of<R> v; //!< The lanes.
    };

    //! \brief `a + b`, lane by lane.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_INLINE fft_pack<R> operator+(fft_pack<R> a,
                                                     fft_pack<R> b) noexcept
    {
      return {lanes_add(a.v, b.v)};
    }

    //! \brief `a − b`, lane by lane.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_INLINE fft_pack<R> operator-(fft_pack<R> a,
                                                     fft_pack<R> b) noexcept
    {
      return {lanes_sub(a.v, b.v)};
    }

    //! \brief `−a`, lane by lane, exact.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_INLINE fft_pack<R> operator-(fft_pack<R> a) noexcept
    {
      return {lanes_negated(a.v)};
    }

    //! \brief `c · a`, each lane's product rounded.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_INLINE fft_pack<R> operator*(R           c,
                                                     fft_pack<R> a) noexcept
    {
      return {lanes_mul(lanes_splat(c), a.v)};
    }

    //! \brief `c · a + b`, each lane one fused multiply-add.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_INLINE fft_pack<R> fused(R           c,
                                                 fft_pack<R> a,
                                                 fft_pack<R> b) noexcept
    {
      return {lanes_fused(lanes_splat(c), a.v, b.v)};
    }

    /*!
     * \brief A register of lines' complex numbers: lane `b` of \ref re and \ref im the real and imaginary parts of line
     *        `b`'s number, so that the passes' arithmetic (\ref twiddled, \ref scaled, \ref scaled_add and the sums)
     *        runs each line's own operations, in its order: a line's bits, among a register of others.
     */
    template <std::floating_point R>
    struct fft_lanes
    {
      using value_type = R; //!< The parts' type.

      fft_pack<R> re;       //!< The real parts.
      fft_pack<R> im;       //!< The imaginary parts.

      //! \brief The real parts.
      [[nodiscard]] NUMERICAL_FFT_LANES_INLINE fft_pack<R> real() const noexcept
      {
        return re;
      }

      //! \brief The imaginary parts.
      [[nodiscard]] NUMERICAL_FFT_LANES_INLINE fft_pack<R> imag() const noexcept
      {
        return im;
      }

      //! \brief Each lane's sum with \p other's.
      NUMERICAL_FFT_LANES_INLINE fft_lanes& operator+=(const fft_lanes& other) noexcept
      {
        re = re + other.re;
        im = im + other.im;
        return *this;
      }
    };

    //! \brief The lines' lanes per register of \p R.
    template <std::floating_point R>
    inline constexpr std::size_t fft_lanes_width {sizeof(fft_lanes_of<R>) / sizeof(R)};

    //! \brief `a + b`, lane by lane.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_INLINE fft_lanes<R> operator+(const fft_lanes<R>& a,
                                                      const fft_lanes<R>& b) noexcept
    {
      return {a.re + b.re, a.im + b.im};
    }

    //! \brief `a − b`, lane by lane.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_INLINE fft_lanes<R> operator-(const fft_lanes<R>& a,
                                                      const fft_lanes<R>& b) noexcept
    {
      return {a.re - b.re, a.im - b.im};
    }

    //! \brief \ref fused_product of a factor \p w and each lane of \p v: the same operations, lane by lane.
    template <bool Fused,
              std::floating_point R>
    requires Fused
    [[nodiscard]] NUMERICAL_FFT_LANES_INLINE fft_lanes<R> twiddled(const std::complex<R>& w,
                                                                   const fft_lanes<R>&    v) noexcept
    {
      const fft_pack<R> ii {w.imag() * v.im};
      const fft_pack<R> ir {w.imag() * v.re};
      return {fused(w.real(), v.re, -ii), fused(w.real(), v.im, ir)};
    }

#endif

    //! \brief The largest prime a Stockham pass takes as its radix, by its own `r`-point sum: a length with a larger
    //!        prime factor is transformed by Bluestein's convolution.
    inline constexpr std::size_t fft_largest_radix {61};

    /*!
     * \brief The radix of each Stockham pass of \p n, or none where \p n has a prime factor past \ref fft_largest_radix,
     *        its powers of two first (4 first, its stride 1, then 16 while 16 points remain, then 8 or 4), then its factors 3,
     *        5 and 7, then each other prime in ascending order, then a lone factor 2 last, where its pass has no factor.
     */
    [[nodiscard]] inline std::vector<std::size_t> fft_radices(std::size_t n)
    {
      std::vector<std::size_t> radices;
      std::size_t              twos {std::size_t {1} << static_cast<unsigned>(std::countr_zero(n))};
      std::size_t              rest {n / twos};
      if (twos >= 4) {
        radices.push_back(4);
        twos /= 4;
      }
      for (; twos >= 16; twos /= 16) {
        radices.push_back(16);
      }
      if (twos == 8) {
        radices.push_back(8); // three factors of 2: one pass, where 4 and 2 took two
        twos = 1;
      }
      for (; twos >= 4; twos /= 4) {
        radices.push_back(4);
      }
      for (std::size_t prime {3}; rest > 1; prime += 2) {
        if (prime > fft_largest_radix) {
          return {}; // Bluestein's
        }
        for (; rest % prime == 0; rest /= prime) {
          radices.push_back(prime);
        }
      }
      if (twos == 2) {
        radices.push_back(2); // last: no factor
      }
      return radices;
    }

    //! \brief The real constants of the radix-3, -5 and -8 butterflies, rounded once from their double-double values.
    template <std::floating_point R>
    struct small_radix_constants
    {
      R half_root_three; //!< √3/2 = sin(2π/3).
      R c1;              //!< cos(2π/5).
      R c2;              //!< cos(4π/5).
      R s1;              //!< sin(2π/5).
      R s2;              //!< sin(4π/5).
      R half_root_two;   //!< √½ = cos(π/4).
    };

    //! \brief \ref small_radix_constants from the roots of order 3 and 5 (\ref unit_root, whose imaginary parts are
    //!        `−sin`).
    template <std::floating_point R>
    [[nodiscard]] small_radix_constants<R> make_small_radix_constants()
    {
      const std::complex<double> third  {unit_root(1, 3)};
      const std::complex<double> fifth  {unit_root(1, 5)};
      const std::complex<double> fifth2 {unit_root(2, 5)};
      return {static_cast<R>(-third.imag()), static_cast<R>(fifth.real()), static_cast<R>(fifth2.real()), static_cast<R>(-fifth.imag()),
              static_cast<R>(-fifth2.imag()), static_cast<R>(unit_root(1, 8).real())};
    }

    template <std::floating_point R>
    struct fft_plan;

    //! \brief Bluestein's convolution for a length \p n with a prime factor past \ref fft_largest_radix.
    template <std::floating_point R>
    struct bluestein_plan
    {
      std::size_t                        padded {0}; //!< M: the power of two at least 2n − 1 the convolution runs at.
      std::vector<std::complex<R>>       chirp;      //!< `e^{−πi j²/n}`, its exponent `j² mod 2n` reduced exactly.
      std::vector<std::complex<R>>       spectrum;   //!< The transform of M points of the chirp's conjugate, wrapped.
      std::shared_ptr<const fft_plan<R>> inner;      //!< The plan of M points.
    };

    /*!
     * \brief The fewest bytes of complex numbers a transform takes in four steps (\ref four_step_plan): fewer stay in
     *        the caches through the passes of one transform, faster on a thread alone (six AVX2 cores: complex128 2¹⁷
     *        points 2.9 ms in passes, 3.2 ms in four steps; 2²⁰ points 21.9 against 17.1 ms, 2²² 208 against 80 ms, all
     *        threads 174 against 48 ms). By the length and the type alone, so that a transform's bits are its own on any
     *        threads.
     */
    inline constexpr std::size_t fft_four_step_least {std::size_t {1} << 24};

    //! \brief The complex numbers past each row of the four steps' working space: rows a power of two of bytes apart
    //!        fall in a few sets of the first-level cache, a column's elements evicting one another.
    inline constexpr std::size_t fft_four_step_pad {8};

    /*!
     * \brief A transform of `n = n1·n2` points in four steps (Bailey's), each a batch of lines the threads share: `x` as
     *        `n2` rows of `n1`, `x[j1 + n1·j2]`; each of its `n1` columns transformed in place (\ref column_plan, `n2`
     *        points, a line's transform as \ref transform_axis's along a first axis); each `(k2, j1)` times `w^{j1·k2}`
     *        (\ref twiddles); each of its `n2` rows transformed (\ref row_plan, `n1` points), `X[k2 + n2·k1]` row
     *        `k2`'s `k1`; transposed into place. Each element's operations fixed by `n` alone: its bits on any threads.
     */
    template <std::floating_point R>
    struct four_step_plan
    {
      std::size_t                        columns {0}; //!< n1: the columns, the points of each row.
      std::size_t                        rows    {0}; //!< n2: the rows, the points of each column.
      std::shared_ptr<const fft_plan<R>> column_plan; //!< The plan of `n2` points.
      std::shared_ptr<const fft_plan<R>> row_plan;    //!< The plan of `n1` points.
      std::vector<std::complex<R>>       twiddles;    //!< `w^{j1·k2}`, `e^{−2πi j1·k2/n}`, at `k2·n1 + j1`.
    };

    //! \brief The factors and passes of a transform of one length, built once and shared.
    template <std::floating_point R>
    struct fft_plan
    {
      std::size_t                       length {0}; //!< The transform's length.
      std::vector<std::size_t>          radices;    //!< Each Stockham pass' radix (\ref fft_radices).
      std::vector<std::complex<R>>      twiddles;
      //!< Each pass' factors from `p` = 1: its `w^p` run, then its `w^{2p}` run, to its `w^{(r−1)p}` run.
      std::vector<std::complex<R>>      radix_roots;
      //!< Each pass of a prime radix past 5: `e^{−2πi m/r}` for m below r, the passes in order.
      std::array<std::complex<R>, 16>   sixteenths; //!< `e^{−2πi k/16}`: a radix-16 butterfly's own factors.
      std::optional<bluestein_plan<R>>  bluestein;  //!< Where the length has a prime factor past fft_largest_radix.
      std::optional<four_step_plan<R>>  four_step;  //!< Where the length is at least fft_four_step_least, in passes.
      std::size_t                       work {0};   //!< The working space a transform takes beside its data.
      small_radix_constants<R>          small;      //!< The radix-3 and radix-5 butterflies' constants.
    };

    template <std::floating_point R>
    void forward_fft(const fft_plan<R>& plan,
                     std::complex<R>*   data,
                     std::complex<R>*   work,
                     bool               fused,
                     bool               in_work = false);

    //! \brief Whether a transform under \p plan, its input placed in its working space (forward_fft's `in_work`), ends in
    //!        its data without a copy: an odd number of Stockham passes, no Bluestein's convolution nor four steps.
    template <std::floating_point R>
    [[nodiscard]] bool fft_starts_in_work(const fft_plan<R>& plan) noexcept
    {
      return !plan.bluestein && !plan.four_step && plan.radices.size() % 2 == 1;
    }

    /*!
     * \brief The columns `n1` of a transform of \p n points of \p bytes each in four steps: the largest divisor of `n`
     *        at most `√n`, `n` of at least \ref fft_four_step_least bytes; else 0, one transform of passes. `n` is taken
     *        in passes (its prime factors at most fft_largest_radix: make_fft_plan settles Bluestein's lengths first). Such an `n`, 2²⁰ and more, has a divisor from
     *        16 to `√n` (the product of its smallest prime factors first past 15 is at most 15·61), so that neither side
     *        is short.
     */
    [[nodiscard]] inline std::size_t four_step_columns(std::size_t n,
                                                       std::size_t bytes)
    {
      if (n < fft_four_step_least / bytes) {
        return 0;
      }
      std::size_t columns {static_cast<std::size_t>(std::sqrt(static_cast<double>(n)))}; // ⌊√n⌋: sqrt rounds correctly, n below 2⁵²
      while (n % columns != 0) {
        --columns;
      }
      return columns;
    }

    //! \brief The \ref fft_plan of length \p n.
    template <std::floating_point R>
    [[nodiscard]] fft_plan<R> make_fft_plan(std::size_t n)
    {
      using C = std::complex<R>;
      const unit_roots roots {n};
      const auto       narrow {[](std::complex<double> value) { return C {static_cast<R>(value.real()), static_cast<R>(value.imag())}; }};
      fft_plan<R>      plan {.length = n, .radices = fft_radices(n), .twiddles = {}, .radix_roots = {}, .sixteenths = {}, .bluestein = {},
                             .four_step = {}, .work = n, .small = make_small_radix_constants<R>()};
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
        std::vector<C> scratch(b.inner->work);
        forward_fft(*b.inner, b.spectrum.data(), scratch.data(), false, false); // the portable arithmetic: the same on every path
        plan.work      = b.padded + b.inner->work;                              // the convolution's sequence, then its transforms' space
        plan.bluestein = std::move(b);
        return plan;
      }
      const std::size_t columns {four_step_columns(n, sizeof(C))};
      if (columns != 0) {
        four_step_plan<R> f {.columns = columns, .rows = n / columns, .column_plan = {}, .row_plan = {}, .twiddles = {}};
        f.column_plan = std::make_shared<const fft_plan<R>>(make_fft_plan<R>(f.rows));
        f.row_plan    = std::make_shared<const fft_plan<R>>(make_fft_plan<R>(f.columns));
        f.twiddles.resize(n);
        for (std::size_t k2 {0}; k2 < f.rows; ++k2) {
          for (std::size_t j1 {0}; j1 < f.columns; ++j1) {
            f.twiddles[(k2 * f.columns) + j1] = narrow(roots.root((j1 * k2) % n));
          }
        }
        plan.radices.clear();
        plan.work      = f.rows * (f.columns + fft_four_step_pad); // the columns' transforms, transposed into place
        plan.four_step = std::move(f);
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
        if (radix > 5 && radix != 8 && radix != 16) {
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

    //! \brief The part type of a complex number (`R`), or of a register of lines' (`fft_pack`).
    template <typename C>
    using fft_real_t = typename C::value_type;

    //! \brief The complex number of \p C's part type: a pass' factors.
    template <typename C>
    using fft_root_t = std::complex<fft_real_t<C>>;

    /*!
     * \brief One Stockham pass of radix 4 over \p in into \p out: \p current points at stride \p stride, so
     *        `current · stride` is the length. Element `q + stride · (p + j · current/4)` of the input is the `j`-th
     *        input of butterfly `(p, q)`, whose outputs go to `q + stride · (4p + j)`, the last three times `w^{jp}`
     *        (\ref twiddled), none for `p` = 0.
     */
    template <bool Fused,
              typename C>
    NUMERICAL_ALWAYS_INLINE void radix_four_pass_body(std::size_t          current,
                                                      std::size_t          stride,
                                                      const C *            in,
                                                      C *                  out,
                                                      const fft_root_t<C>* twiddles) noexcept
    {
      const std::size_t quarter {current / 4};
      const std::size_t run     {quarter - 1}; // the factors of each power
      const std::size_t apart   {stride * quarter};
      for (std::size_t p {0}; p < quarter; ++p) {
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
          const C first       {a_minus_c - i_b_minus_d};
          const C second      {a_plus_c - b_plus_d};
          const C third       {a_minus_c + i_b_minus_d};
          to[q]              = a_plus_c + b_plus_d;
          to[q + stride]     = p == 0 ? first : twiddled<Fused>(twiddles[p - 1], first);
          to[q + 2 * stride] = p == 0 ? second : twiddled<Fused>(twiddles[run + p - 1], second);
          to[q + 3 * stride] = p == 0 ? third : twiddled<Fused>(twiddles[(2 * run) + p - 1], third);
        }
      }
    }

    //! \brief \ref radix_four_pass_body of complex numbers, its products as \p Fused.
    template <bool Fused,
              std::floating_point R>
    void radix_four_pass(std::size_t            current,
                         std::size_t            stride,
                         const std::complex<R>* in,
                         std::complex<R>*       out,
                         const std::complex<R>* twiddles) noexcept
    {
      radix_four_pass_body<Fused>(current, stride, in, out, twiddles);
    }

    //! \brief A 4-point transform of \p v0 to \p v3 into \p y: the radix-4 butterfly's sums, in its order.
    template <typename C>
    NUMERICAL_ALWAYS_INLINE void four_point(const C& v0,
                                            const C& v1,
                                            const C& v2,
                                            const C& v3,
                                            C*       y) noexcept
    {
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
              typename C>
    NUMERICAL_ALWAYS_INLINE void radix_sixteen_pass_body(std::size_t                          current,
                                                         std::size_t                          stride,
                                                         const C*                             in,
                                                         C*                                   out,
                                                         const fft_root_t<C>*                 twiddles,
                                                         const std::array<fft_root_t<C>, 16>& sixteenths) noexcept
    {
      const std::size_t part  {current / 16};
      const std::size_t run   {part - 1};
      const std::size_t apart {stride * part};
      for (std::size_t p {0}; p < part; ++p) {
        const C* const in_p {in + (stride * p)};
        C* const       to   {out + (stride * 16 * p)};
        for (std::size_t q {0}; q < stride; ++q) {
          std::array<C, 16> inner {};
          NUMERICAL_UNROLL_BLOCK
          for (std::size_t j1 {0}; j1 < 4; ++j1) {
            four_point(in_p[q + (j1 * apart)], in_p[q + ((j1 + 4) * apart)], in_p[q + ((j1 + 8) * apart)], in_p[q + ((j1 + 12) * apart)],
                       inner.data() + (4 * j1));
            NUMERICAL_UNROLL_BLOCK
            for (std::size_t k2 {1}; k2 < 4 && j1 != 0; ++k2) {
              C& value {inner[(4 * j1) + k2]};
              value = (j1 * k2) == 4 ? C {value.imag(), -value.real()} : twiddled<Fused>(sixteenths[j1 * k2], value);
            }
          }
          NUMERICAL_UNROLL_BLOCK
          for (std::size_t k2 {0}; k2 < 4; ++k2) {
            std::array<C, 4> outputs {};
            four_point(inner[k2], inner[4 + k2], inner[8 + k2], inner[12 + k2], outputs.data());
            NUMERICAL_UNROLL_BLOCK
            for (std::size_t k1 {0}; k1 < 4; ++k1) {
              const std::size_t k {(4 * k1) + k2};
              to[q + (stride * k)] = (p == 0 || k == 0) ? outputs[k1] : twiddled<Fused>(twiddles[((k - 1) * run) + p - 1], outputs[k1]);
            }
          }
        }
      }
    }

    //! \brief \ref radix_sixteen_pass_body of complex numbers, its products as \p Fused.
    template <bool Fused,
              std::floating_point R>
    void radix_sixteen_pass(std::size_t                            current,
                            std::size_t                            stride,
                            const std::complex<R>*                 in,
                            std::complex<R>*                       out,
                            const std::complex<R>*                 twiddles,
                            const std::array<std::complex<R>, 16>& sixteenths) noexcept
    {
      radix_sixteen_pass_body<Fused>(current, stride, in, out, twiddles, sixteenths);
    }

    //! \brief `c · v` of a real \p c and a complex \p v: each part's product, rounded.
    template <std::floating_point R>
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> scaled(R                      c,
                                                                 const std::complex<R>& v) noexcept
    {
      return {c * v.real(), c * v.imag()};
    }

    //! \brief `s + c · v` of a complex \p s, a real \p c and a complex \p v on the vector paths: each part one fused
    //!        multiply-add.
    template <bool Fused,
              std::floating_point R>
    requires Fused
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> scaled_add(const std::complex<R>& s,
                                                                     R                      c,
                                                                     const std::complex<R>& v) noexcept
    {
      return {std::fma(c, v.real(), s.real()), std::fma(c, v.imag(), s.imag())};
    }

    //! \brief \ref scaled_add on the portable path: the product rounded first.
    template <bool Fused,
              std::floating_point R>
    requires (!Fused)
    [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> scaled_add(const std::complex<R>& s,
                                                                     R                      c,
                                                                     const std::complex<R>& v) noexcept
    {
      return s + scaled(c, v);
    }

#ifdef NUMERICAL_FFT_LANES
    //! \brief \ref scaled of each lane of \p v: each part's product, rounded.
    template <std::floating_point R>
    [[nodiscard]] NUMERICAL_FFT_LANES_INLINE fft_lanes<R> scaled(R                   c,
                                                                 const fft_lanes<R>& v) noexcept
    {
      return {c * v.re, c * v.im};
    }

    //! \brief \ref scaled_add of each lane on the vector paths: each part one fused multiply-add.
    template <bool Fused,
              std::floating_point R>
    requires Fused
    [[nodiscard]] NUMERICAL_FFT_LANES_INLINE fft_lanes<R> scaled_add(const fft_lanes<R>& s,
                                                                     R                   c,
                                                                     const fft_lanes<R>& v) noexcept
    {
      return {fused(c, v.re, s.re), fused(c, v.im, s.im)};
    }

#endif
    /*!
     * \brief One Stockham pass of an odd radix \p radix, as \ref radix_four_pass is of 4, by the symmetry of its roots
     *        (\ref radix_five_pass_body's, any odd `r`): `a_j = x_j + x_{r−j}`, `b_j = x_j − x_{r−j}` (`j` from 1 to
     *        `h = (r − 1)/2`); `X_0 = x_0 + a_1 + … + a_h`, and for `k` from 1 to `h`, `A_k = x_0 + Σ_j cos(2πjk/r)·a_j`,
     *        `B_k = Σ_j sin(2πjk/r)·b_j`, each sum in ascending `j`, its products fused as \p Fused (`scaled_add`;
     *        the cosines and sines \p roots' parts, `jk` reduced mod `r` by additions), `X_k = A_k − i·B_k`,
     *        `X_{r−k} = A_k + i·B_k`; output `k` then times `w^{kp}`, none for `p` = 0. Radices 3 and 5 so computed give
     *        their own butterflies' bits. Inlined into each caller, so that the x86-64 fused one is built for FMA.
     */
    template <bool Fused,
              typename C>
    NUMERICAL_ALWAYS_INLINE void any_radix_pass_body(std::size_t          radix,
                                                     std::size_t          current,
                                                     std::size_t          stride,
                                                     const C*             in,
                                                     C*                   out,
                                                     const fft_root_t<C>* twiddles,
                                                     const fft_root_t<C>* roots) noexcept
    {
      const std::size_t part  {current / radix};
      const std::size_t run   {part - 1};
      const std::size_t apart {stride * part};
      const std::size_t half  {radix / 2};
      for (std::size_t p {0}; p < part; ++p) {
        const C* const in_p {in + (stride * p)};
        C* const       to   {out + (stride * radix * p)};
        for (std::size_t q {0}; q < stride; ++q) {
          std::array<C, fft_largest_radix / 2> sums        {};
          std::array<C, fft_largest_radix / 2> differences {};
          const C                              x0          {in_p[q]};
          C                                    total       {x0};
          for (std::size_t j {1}; j <= half; ++j) {
            const C x {in_p[q + (j * apart)]};
            const C y {in_p[q + ((radix - j) * apart)]};
            sums[j - 1]        = x + y;
            differences[j - 1] = x - y;
            total              = total + sums[j - 1];
          }
          to[q] = total;
          for (std::size_t k {1}; k <= half; ++k) {
            std::size_t power      {k}; // j·k mod r, kept by additions
            C           cosines    {scaled_add<Fused>(x0, roots[power].real(), sums[0])};
            C           sines      {scaled(-roots[power].imag(), differences[0])};
            for (std::size_t j {2}; j <= half; ++j) {
              power   += k;
              power   -= power >= radix ? radix : 0;
              cosines  = scaled_add<Fused>(cosines, roots[power].real(), sums[j - 1]);
              sines    = scaled_add<Fused>(sines, -roots[power].imag(), differences[j - 1]);
            }
            const C i_sines {-sines.imag(), sines.real()}; // i·B_k, exact
            const C low     {cosines - i_sines};
            const C high    {cosines + i_sines};
            to[q + (stride * k)]           = p == 0 ? low : twiddled<Fused>(twiddles[((k - 1) * run) + p - 1], low);
            to[q + (stride * (radix - k))] = p == 0 ? high : twiddled<Fused>(twiddles[((radix - k - 1) * run) + p - 1], high);
          }
        }
      }
    }

    /*!
     * \brief One Stockham pass of radix 3, as \ref any_radix_pass_body is of any: `t = x1 + x2`, `v = x0 − t/2` (exact
     *        halving, one rounding), `u = (√3/2)·(x1 − x2)`; `X0 = x0 + t`, `X1 = v − i·u`, `X2 = v + i·u`, then times
     *        `w^{kp}` as \p Fused. Inlined into each caller.
     */
    template <bool Fused,
              typename C>
    NUMERICAL_ALWAYS_INLINE void radix_three_pass_body(std::size_t                                 current,
                                                       std::size_t                                 stride,
                                                       const C           *                         in,
                                                       C           *                               out,
                                                       const fft_root_t<C>           *             twiddles,
                                                       const small_radix_constants<fft_real_t<C>>& k) noexcept
    {
      using R = fft_real_t<C>;
      const std::size_t part  {current / 3};
      const std::size_t run   {part - 1};
      const std::size_t apart {stride * part};
      for (std::size_t p {0}; p < part; ++p) {
        const C* const in_p {in + (stride * p)};
        C* const       to   {out + (stride * 3 * p)};
        for (std::size_t q {0}; q < stride; ++q) {
          const C x0 {in_p[q]};
          const C x1 {in_p[q + apart]};
          const C x2 {in_p[q + (2 * apart)]};
          const C t  {x1 + x2};
          const C v  {x0 - scaled(R(0.5), t)};
          const C u  {scaled(k.half_root_three, x1 - x2)};
          const C iu {-u.imag(), u.real()};
          const C y1 {v - iu};
          const C y2 {v + iu};
          to[q]              = x0 + t;
          to[q + stride]     = p == 0 ? y1 : twiddled<Fused>(twiddles[p - 1], y1);
          to[q + 2 * stride] = p == 0 ? y2 : twiddled<Fused>(twiddles[run + p - 1], y2);
        }
      }
    }

    /*!
     * \brief One Stockham pass of radix 5: `t1 = x1 + x4`, `t2 = x2 + x3`, `t3 = x1 − x4`, `t4 = x2 − x3`;
     *        `a1 = x0 + c1·t1 + c2·t2`, `a2 = x0 + c2·t1 + c1·t2`, `b1 = s1·t3 + s2·t4`, `b2 = s2·t3 − s1·t4` (each
     *        sum in that order, its products fused as \p Fused); `X0 = x0 + t1 + t2`, `X1 = a1 − i·b1`, `X4 = a1 + i·b1`,
     *        `X2 = a2 − i·b2`, `X3 = a2 + i·b2`, then times `w^{kp}`. Inlined into each caller.
     */
    template <bool Fused,
              typename C>
    NUMERICAL_ALWAYS_INLINE void radix_five_pass_body(std::size_t                                 current,
                                                      std::size_t                                 stride,
                                                      const C           *                         in,
                                                      C           *                               out,
                                                      const fft_root_t<C>           *             twiddles,
                                                      const small_radix_constants<fft_real_t<C>>& k) noexcept
    {
      const std::size_t part  {current / 5};
      const std::size_t run   {part - 1};
      const std::size_t apart {stride * part};
      for (std::size_t p {0}; p < part; ++p) {
        const C* const in_p {in + (stride * p)};
        C* const       to   {out + (stride * 5 * p)};
        for (std::size_t q {0}; q < stride; ++q) {
          const C                x0  {in_p[q]};
          const C                x1  {in_p[q + apart]};
          const C                x2  {in_p[q + (2 * apart)]};
          const C                x3  {in_p[q + (3 * apart)]};
          const C                x4  {in_p[q + (4 * apart)]};
          const C                t1  {x1 + x4};
          const C                t2  {x2 + x3};
          const C                t3  {x1 - x4};
          const C                t4  {x2 - x3};
          const C                a1  {scaled_add<Fused>(scaled_add<Fused>(x0, k.c1, t1), k.c2, t2)};
          const C                a2  {scaled_add<Fused>(scaled_add<Fused>(x0, k.c2, t1), k.c1, t2)};
          const C                b1  {scaled_add<Fused>(scaled(k.s1, t3), k.s2, t4)};
          const C                b2  {scaled_add<Fused>(scaled(k.s2, t3), -k.s1, t4)};
          const C                ib1 {-b1.imag(), b1.real()};
          const C                ib2 {-b2.imag(), b2.real()};
          const std::array<C, 5> y   {(x0 + t1) + t2, a1 - ib1, a2 - ib2, a2 + ib2, a1 + ib1};
          to[q] = y[0];
          for (std::size_t j {1}; j < 5; ++j) {
            to[q + (j * stride)] = p == 0 ? y[j] : twiddled<Fused>(twiddles[((j - 1) * run) + p - 1], y[j]);
          }
        }
      }
    }

    /*!
     * \brief One Stockham pass of radix 8, as \ref radix_four_pass is of 4: `a_j = x_j + x_{j+4}`, `b_j = x_j − x_{j+4}`
     *        (j below 4), `b_j` then times `e^{−2πi j/8}` (`√½·(re + im, im − re)`, `(im, −re)`,
     *        `√½·(im − re, −(re + im))`: each sum then its product, no fused operation, so the same bits on every
     *        path); the even outputs `a`'s \ref four_point, the odd ones `b`'s; output `k` then times `w^{kp}` as
     *        \p Fused, none for `p` = 0. \p half_root_two is √½.
     */
    template <bool Fused,
              typename C>
    NUMERICAL_ALWAYS_INLINE void radix_eight_pass_body(std::size_t          current,
                                                       std::size_t          stride,
                                                       const C*             in,
                                                       C*                   out,
                                                       const fft_root_t<C>* twiddles,
                                                       fft_real_t<C>        half_root_two) noexcept
    {
      const std::size_t part  {current / 8};
      const std::size_t run   {part - 1};
      const std::size_t apart {stride * part};
      for (std::size_t p {0}; p < part; ++p) {
        const C* const in_p {in + (stride * p)};
        C* const       to   {out + (stride * 8 * p)};
        for (std::size_t q {0}; q < stride; ++q) {
          std::array<C, 4> a {};
          std::array<C, 4> b {};
          for (std::size_t j {0}; j < 4; ++j) {
            const C low  {in_p[q + (j * apart)]};
            const C high {in_p[q + ((j + 4) * apart)]};
            a[j] = low + high;
            b[j] = low - high;
          }
          const auto sum1        {b[1].real() + b[1].imag()};
          const auto difference1 {b[1].imag() - b[1].real()};
          const auto sum3        {b[3].real() + b[3].imag()};
          const auto difference3 {b[3].imag() - b[3].real()};
          b[1] = C {half_root_two * sum1, half_root_two * difference1};
          b[2] = C {b[2].imag(), -b[2].real()};
          b[3] = C {half_root_two * difference3, -(half_root_two * sum3)};
          std::array<C, 4> even {};
          std::array<C, 4> odd  {};
          four_point(a[0], a[1], a[2], a[3], even.data());
          four_point(b[0], b[1], b[2], b[3], odd.data());
          for (std::size_t k {0}; k < 8; ++k) {
            const C value {(k % 2) == 0 ? even[k / 2] : odd[k / 2]};
            to[q + (stride * k)] = (p == 0 || k == 0) ? value : twiddled<Fused>(twiddles[((k - 1) * run) + p - 1], value);
          }
        }
      }
    }

    //! \brief \ref any_radix_pass_body, its products as \p Fused.
    template <bool Fused,
              std::floating_point R>
    void any_radix_pass(std::size_t                     radix,
                        std::size_t                     current,
                        std::size_t                     stride,
                        const std::complex<R>         * in,
                        std::complex<R>         *       out,
                        const std::complex<R>         * twiddles,
                        const std::complex<R>         * roots,
                        const small_radix_constants<R>& small) noexcept
    {
      if (radix == 3) {
        radix_three_pass_body<Fused>(current, stride, in, out, twiddles, small);
      }
      else if (radix == 5) {
        radix_five_pass_body<Fused>(current, stride, in, out, twiddles, small);
      }
      else if (radix == 8) {
        radix_eight_pass_body<Fused>(current, stride, in, out, twiddles, small.half_root_two);
      }
      else {
        any_radix_pass_body<Fused>(radix, current, stride, in, out, twiddles, roots);
      }
    }

#ifdef NUMERICAL_X86_COMPILED
    //! \brief \ref any_radix_pass_body fused, built for AVX2 with FMA: each `std::fma` one instruction.
    template <std::floating_point R>
    NUMERICAL_OUT_OF_LINE NUMERICAL_TARGET("avx2,fma") void any_radix_pass_x86(std::size_t                     radix,
                                                                               std::size_t                     current,
                                                                               std::size_t                     stride,
                                                                               const std::complex<R>         * in,
                                                                               std::complex<R>         *       out,
                                                                               const std::complex<R>         * twiddles,
                                                                               const std::complex<R>         * roots,
                                                                               const small_radix_constants<R>& small) noexcept
    {
      if (radix == 3) {
        avx2_radix_three_pass(current, stride, in, out, twiddles, small.half_root_three);
      }
      else if (radix == 5) {
        avx2_radix_five_pass(current, stride, in, out, twiddles, small.c1, small.c2, small.s1, small.s2);
      }
      else if (radix == 8) {
        avx2_radix_eight_pass(current, stride, in, out, twiddles, small.half_root_two);
      }
      else {
        any_radix_pass_body<true>(radix, current, stride, in, out, twiddles, roots);
      }
    }

    //! \brief \ref any_radix_pass_x86 of the avx512 path: radices 3, 5 and 8 in AVX-512's registers where the stride
    //!        holds them, else \ref any_radix_pass_x86 itself (out of line: inlined here, its code built for AVX-512 ran
    //!        up to a fifth slower, ten AVX-512 cores, complex64 5⁶ points).
    template <std::floating_point R>
    NUMERICAL_TARGET("avx512f,avx2,fma") void any_radix_pass_avx512(std::size_t                     radix,
                                                                    std::size_t                     current,
                                                                    std::size_t                     stride,
                                                                    const std::complex<R>         * in,
                                                                    std::complex<R>         *       out,
                                                                    const std::complex<R>         * twiddles,
                                                                    const std::complex<R>         * roots,
                                                                    const small_radix_constants<R>& small) noexcept
    {
      if (stride % (sizeof(fft_reg<fft_avx512, R>) / sizeof(std::complex<R>)) != 0) {
        any_radix_pass_x86(radix, current, stride, in, out, twiddles, roots, small);
      }
      else if (radix == 3) {
        avx512_radix_three_pass(current, stride, in, out, twiddles, small.half_root_three);
      }
      else if (radix == 5) {
        avx512_radix_five_pass(current, stride, in, out, twiddles, small.c1, small.c2, small.s1, small.s2);
      }
      else if (radix == 8) {
        avx512_radix_eight_pass(current, stride, in, out, twiddles, small.half_root_two);
      }
      else {
        any_radix_pass_body<true>(radix, current, stride, in, out, twiddles, roots);
      }
    }

#endif

#ifdef NUMERICAL_NEON_COMPILED
    //! \brief \ref any_radix_pass fused, radices 3, 5 and 8 in NEON's registers.
    template <std::floating_point R>
    void any_radix_pass_neon(std::size_t                     radix,
                             std::size_t                     current,
                             std::size_t                     stride,
                             const std::complex<R>         * in,
                             std::complex<R>         *       out,
                             const std::complex<R>         * twiddles,
                             const std::complex<R>         * roots,
                             const small_radix_constants<R>& small) noexcept
    {
      if (radix == 3) {
        neon_radix_three_pass(current, stride, in, out, twiddles, small.half_root_three);
      }
      else if (radix == 5) {
        neon_radix_five_pass(current, stride, in, out, twiddles, small.c1, small.c2, small.s1, small.s2);
      }
      else if (radix == 8) {
        neon_radix_eight_pass(current, stride, in, out, twiddles, small.half_root_two);
      }
      else {
        any_radix_pass_body<true>(radix, current, stride, in, out, twiddles, roots);
      }
    }

#endif

    //! \brief The last pass of an odd power of two: butterflies of radix 2 at stride \p stride, no factor.
    template <typename C>
    NUMERICAL_ALWAYS_INLINE void radix_two_pass(std::size_t stride,
                                                const C*    in,
                                                C*          out) noexcept
    {
      for (std::size_t q {0}; q < stride; ++q) {
        const C a {in[q]};
        const C b {in[q + stride]};
        out[q]          = a + b;
        out[q + stride] = a - b;
      }
    }

    /*!
     * \brief The Stockham passes of the \p plan over \p data, \p work as long, the result in \p data: radix 4 by \p four
     *        (\ref radix_four_pass's arguments), 16 by \p sixteen (\ref radix_sixteen_pass's), 2 by \ref radix_two_pass
     *        (always the last pass), any other by \p other (\ref any_radix_pass's: 3 and 5 by their own butterflies).
     *        The input in \p data, or \p from_work in \p work (`fft_starts_in_work`: an odd number of passes then
     *        ends in \p data, where from \p data the last pass wrote \p work, copied back).
     */
    template <std::floating_point R,
              typename C,
              typename Four,
              typename Sixteen,
              typename Other>
    NUMERICAL_ALWAYS_INLINE void stockham_passes(const fft_plan<R>& plan,
                                                 C*                 data,
                                                 C*                 work,
                                                 Four               four,
                                                 Sixteen            sixteen,
                                                 Other              other,
                                                 bool               from_work = false) noexcept
    {
      const std::size_t      n        {plan.length};
      C*                     in       {from_work ? work : data};
      C*                     out      {from_work ? data : work};
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
        else if (radix == 2) { // always the last pass (fft_radices): no factor
          radix_two_pass(stride, in, out);
        }
        else {
          other(radix, current, stride, in, out, twiddles, roots, plan.small);
          roots += (radix > 5 && radix != 8) ? radix : 0; // 3, 5 and 8 take the plan's constants
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
     * \brief Bluestein's transform of the \p plan's length in place in \p data, \p work its \ref fft_plan::work, the
     *        input times the chirp, padded to M; its transform times the chirp's spectrum; the inverse transform of that
     *        (`conj(fft(conj ·))`, then 1/M, exact), times the chirp. Products as \p Fused. Inlined into each caller.
     */
    template <bool Fused,
              std::floating_point R>
    NUMERICAL_ALWAYS_INLINE void bluestein_body(const fft_plan<R>& plan,
                                                std::complex<R>*   data,
                                                std::complex<R>*   work)
    {
      using C = std::complex<R>;
      const bluestein_plan<R>& b     {*plan.bluestein};
      const std::size_t        n     {plan.length};
      const std::size_t        m     {b.padded};
      const R                  scale {R(1) / static_cast<R>(m)};
      C* const                 a     {work};
      const bool               ahead {fft_starts_in_work(*b.inner)};
      C* const                 input {ahead ? work + m : a}; // where the inner transforms read
      for (std::size_t j {0}; j < n; ++j) {
        input[j] = twiddled<Fused>(b.chirp[j], data[j]);
      }
      std::fill(input + n, input + m, C {});
      forward_fft(*b.inner, a, work + m, Fused, ahead);
      for (std::size_t k {0}; k < m; ++k) {
        input[k] = std::conj(twiddled<Fused>(b.spectrum[k], a[k]));
      }
      forward_fft(*b.inner, a, work + m, Fused, ahead);
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
                                                    std::complex<R>*   work)
    {
      bluestein_body<true>(plan, data, work);
    }

#endif

#ifdef NUMERICAL_X86_COMPILED
    //! \brief The shortest transform the avx512 path takes in AVX-512 registers: shorter ones run faster in AVX2's
    //!        (ten AVX-512 cores: 16 and 64 points 0.06 → 0.08 µs in AVX-512's, from 256 points AVX-512's faster), the
    //!        same bits either way.
    inline constexpr std::size_t fft_avx512_shortest {256};

#endif
    //! \brief `t[i] = w[i] · t[i]` for \p count elements, as \p Fused (\ref twiddled). Inlined into each caller.
    template <bool Fused,
              std::floating_point R>
    NUMERICAL_ALWAYS_INLINE void twiddle_run_body(std::complex<R>*       t,
                                                  const std::complex<R>* w,
                                                  std::size_t            count) noexcept
    {
      for (std::size_t i {0}; i < count; ++i) {
        t[i] = twiddled<Fused>(w[i], t[i]);
      }
    }

#ifdef NUMERICAL_X86_COMPILED
    //! \brief \ref twiddle_run_body fused, built for AVX2 with FMA: each `std::fma` one instruction.
    template <std::floating_point R>
    NUMERICAL_TARGET("avx2,fma") void twiddle_run_x86(std::complex<R>*       t,
                                                      const std::complex<R>* w,
                                                      std::size_t            count) noexcept
    {
      twiddle_run_body<true>(t, w, count);
    }

#endif

    //! \brief \ref twiddle_run_body on the path in use.
    template <std::floating_point R>
    void twiddle_run(std::complex<R>*       t,
                     const std::complex<R>* w,
                     std::size_t            count,
                     bool                   fused) noexcept
    {
      if (!fused) {
        twiddle_run_body<false>(t, w, count);
        return;
      }
#ifdef NUMERICAL_X86_COMPILED
      twiddle_run_x86(t, w, count);
#else
      twiddle_run_body<true>(t, w, count);
#endif
    }

    /*!
     * \brief The transform of the \p plan's \ref four_step_plan in place in \p data, \p work its \ref fft_plan::work
     *        (`n`: the columns' transforms, transposed back into \p data), the products fused where \p fused.
     */
    template <std::floating_point R>
    void four_step_fft(const fft_plan<R>& plan,
                       std::complex<R>*   data,
                       std::complex<R>*   work,
                       bool               fused);

    /*!
     * \brief The forward transform of the \p plan's length into \p data, \p work its \ref fft_plan::work, its
     *        Stockham passes, or Bluestein's convolution, or its four steps, the products fused where \p fused
     *        (\ref fft_fuses): the input in \p data, or in \p work where \p in_work (only where
     *        \ref fft_starts_in_work).
     */
    template <std::floating_point R>
    void forward_fft(const fft_plan<R>& plan,
                     std::complex<R>*   data,
                     std::complex<R>*   work,
                     bool               fused,
                     bool               in_work)
    {
      if (plan.four_step) {
        four_step_fft(plan, data, work, fused);
        return;
      }
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
        stockham_passes(plan, data, work, radix_four_pass<false, R>, radix_sixteen_pass<false, R>, any_radix_pass<false, R>, in_work);
        return;
      }
#ifdef NUMERICAL_X86_COMPILED
      if (gemm_path_current() == gemm_path::avx512 && plan.length >= fft_avx512_shortest) {
        stockham_passes(plan, data, work, avx512_radix_four_pass<R>, avx512_radix_sixteen_pass<R>, any_radix_pass_avx512<R>, in_work);
        return;
      }
      stockham_passes(plan, data, work, avx2_radix_four_pass<R>, avx2_radix_sixteen_pass<R>, any_radix_pass_x86<R>, in_work);
#elif defined(NUMERICAL_NEON_COMPILED)
      stockham_passes(plan, data, work, neon_radix_four_pass<R>, neon_radix_sixteen_pass<R>, any_radix_pass_neon<R>, in_work);
#else
      stockham_passes(plan, data, work, radix_four_pass<true, R>, radix_sixteen_pass<true, R>, any_radix_pass<true, R>, in_work);
#endif
    }

#ifdef NUMERICAL_FFT_LANES
    //! \brief \ref radix_four_pass_body of registers of lines, fused.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_TARGET void lanes_four_pass(std::size_t             current,
                                                    std::size_t             stride,
                                                    const fft_lanes<R>*     in,
                                                    fft_lanes<R>*           out,
                                                    const std::complex<R>*  twiddles) noexcept
    {
      radix_four_pass_body<true>(current, stride, in, out, twiddles);
    }

    //! \brief \ref radix_sixteen_pass_body of registers of lines, fused.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_TARGET void lanes_sixteen_pass(std::size_t                            current,
                                                       std::size_t                            stride,
                                                       const fft_lanes<R>*                    in,
                                                       fft_lanes<R>*                          out,
                                                       const std::complex<R>*                 twiddles,
                                                       const std::array<std::complex<R>, 16>& sixteenths) noexcept
    {
      radix_sixteen_pass_body<true>(current, stride, in, out, twiddles, sixteenths);
    }

    //! \brief \ref any_radix_pass of registers of lines, fused: 3, 5 and 8 by their own butterflies.
    template <std::floating_point R>
    NUMERICAL_FFT_LANES_TARGET void lanes_any_radix_pass(std::size_t                     radix,
                                                         std::size_t                     current,
                                                         std::size_t                     stride,
                                                         const fft_lanes<R>*             in,
                                                         fft_lanes<R>*                   out,
                                                         const std::complex<R>*          twiddles,
                                                         const std::complex<R>*          roots,
                                                         const small_radix_constants<R>& small) noexcept
    {
      if (radix == 3) {
        radix_three_pass_body<true>(current, stride, in, out, twiddles, small);
      }
      else if (radix == 5) {
        radix_five_pass_body<true>(current, stride, in, out, twiddles, small);
      }
      else if (radix == 8) {
        radix_eight_pass_body<true>(current, stride, in, out, twiddles, small.half_root_two);
      }
      else {
        any_radix_pass_body<true>(radix, current, stride, in, out, twiddles, roots);
      }
    }

    /*!
     * \brief The forward transforms of a register of lines (\ref fft_lanes_width of them), each fused as
     *        \ref forward_fft fuses a line alone, so each line's bits: element `k` of line `b` from
     *        `in[b · in_line + k · in_step]` (the first \p kept, zeros past them; conjugated where \p inverse) into
     *        `out[b · out_line + k · out_step]` (conjugated back where \p inverse, then times \p scale unless 1).
     *        \p lines and \p work hold the plan's length of registers each; the plan has no Bluestein's convolution.
     */
    template <typename T,
              std::floating_point R>
    NUMERICAL_FFT_LANES_TARGET void lanes_transform(const fft_plan<R>& plan,
                                                    const T*           in,
                                                    std::size_t        in_line,
                                                    std::size_t        in_step,
                                                    std::size_t        kept,
                                                    std::complex<R>*   out,
                                                    std::size_t        out_line,
                                                    std::size_t        out_step,
                                                    bool               inverse,
                                                    R                  scale,
                                                    fft_lanes<R>*      lines,
                                                    fft_lanes<R>*      work) noexcept
    {
      const std::size_t   points {plan.length};
      const bool          ahead  {fft_starts_in_work(plan)};
      fft_lanes<R>* const start  {ahead ? work : lines}; // where the passes read the lines: they end in `lines`
      for (std::size_t k {0}; k < points; ++k) {
        fft_lanes<R> value {};
        if (k < kept) {
          lanes_read(in + (k * in_step), in_line, value.re.v, value.im.v);
        }
        start[k] = {value.re, inverse ? -value.im : value.im};
      }
      stockham_passes(plan, lines, work, lanes_four_pass<R>, lanes_sixteen_pass<R>, lanes_any_radix_pass<R>, ahead);
      for (std::size_t k {0}; k < points; ++k) {
        const fft_lanes<R> value {lines[k].re, inverse ? -lines[k].im : lines[k].im};
        if (scale == R(1)) {
          lanes_write(out + (k * out_step), out_line, value.re.v, value.im.v);
        }
        else {
          lanes_write(out + (k * out_step), out_line, (scale * value.re).v, (scale * value.im).v);
        }
      }
    }

#endif

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

    //! \brief The types \ref numerical, :fft transforms: `float`, `double` and their complex numbers.
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
    //!        \ref fft_scratch_kept elements; past it, its own allocation. Each \p Slot is a space of its own, for
    //!        spaces of one type in use at once.
    template <typename C,
              int Slot = 0>
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

#ifdef NUMERICAL_FFT_LANES
    //! \brief The lines a register holds (\ref fft_lanes_width).
    template <std::floating_point R>
    inline constexpr std::size_t fft_lines_width {fft_lanes_width<R>};

    //! \brief The longest lines transformed a register of them at a time: past it, a line alone holds its passes in
    //!        the caches.
    inline constexpr std::size_t fft_lanes_longest {std::size_t {1} << 12};

#  if defined(NUMERICAL_X86_COMPILED)
    //! \brief The most bytes of registers of lines that lie together (along the last axis) transformed at a time: past
    //!        them the two spaces leave the first-level cache, where the AVX2 passes along a line alone hold it.
    inline constexpr std::size_t fft_lanes_together_bytes {std::size_t {1} << 14};
#  else
    //! \brief The most bytes of registers of lines that lie together (along the last axis) transformed at a time.
    inline constexpr std::size_t fft_lanes_together_bytes {std::size_t {1} << 17};
#  endif

    /*!
     * \brief The registers of lines (\ref fft_lines_width each) a batch of \p lines transforms together under \p plan:
     *        on the vector paths (\p fused) of a length without Bluestein's convolution up to \ref fft_lanes_longest,
     *        and, where each line lies \p together (along the last axis), whose registers take at most
     *        \ref fft_lanes_together_bytes; else none.
     */
    template <std::floating_point R>
    [[nodiscard]] std::size_t fft_lane_groups(const fft_plan<R>& plan,
                                              bool               fused,
                                              std::size_t        lines,
                                              bool               together) noexcept
    {
      const bool fits {together ? plan.length * sizeof(fft_lanes<R>) <= fft_lanes_together_bytes : plan.length <= fft_lanes_longest};
      return fused && !plan.bluestein && fits ? lines / fft_lines_width<R> : 0;
    }

    //! \brief \ref lanes_transform of the lines from \p in into \p out, in this thread's working space.
    template <typename T,
              std::floating_point R>
    void fft_lines_together(const fft_plan<R>& plan,
                            const T*           in,
                            std::size_t        in_line,
                            std::size_t        in_step,
                            std::size_t        kept,
                            std::complex<R>*   out,
                            std::size_t        out_line,
                            std::size_t        out_step,
                            bool               inverse,
                            R                  scale)
    {
      const fft_scratch<fft_lanes<R>> space {2 * plan.length};
      lanes_transform(plan, in, in_line, in_step, kept, out, out_line, out_step, inverse, scale, space.data(),
                      space.data() + plan.length);
    }

#else
    //! \brief One line at a time where no register holds lines.
    template <std::floating_point R>
    inline constexpr std::size_t fft_lines_width {1};

    //! \brief No registers of lines.
    template <std::floating_point R>
    [[nodiscard]] std::size_t fft_lane_groups(const fft_plan<R>& /*plan*/,
                                              bool /*fused*/,
                                              std::size_t /*lines*/,
                                              bool /*together*/) noexcept
    {
      return 0;
    }

    //! \brief Never called: no registers of lines.
    template <typename T,
              std::floating_point R>
    void fft_lines_together(const fft_plan<R>& /*plan*/,
                            const T* /*in*/,
                            std::size_t /*in_line*/,
                            std::size_t /*in_step*/,
                            std::size_t /*kept*/,
                            std::complex<R>* /*out*/,
                            std::size_t /*out_line*/,
                            std::size_t /*out_step*/,
                            bool /*inverse*/,
                            R /*scale*/) noexcept
    {
    }

#endif

    //! \brief The lines a task of \ref transform_axis takes together along an axis whose lines lie apart: each row read
    //!        and written as that many neighbours, where one line at a time took one element of each row.
    inline constexpr std::size_t fft_column_block {8};

    //! \brief \p from, \p rows rows of \p columns (\p apart from one another), transposed into \p to (\p columns rows of
    //!        \p rows), by tiles that the threads share.
    template <std::floating_point R>
    void four_step_transpose(const std::complex<R>* from,
                             std::size_t            rows,
                             std::size_t            columns,
                             std::size_t            apart,
                             std::complex<R>*       to)
    {
      const std::size_t bands {(rows + transpose_tile - 1) / transpose_tile};
      const std::size_t piece {transpose_piece(apart * sizeof(std::complex<R>), rows * sizeof(std::complex<R>))};
      parallel_for(bands, transpose_tile * columns, [&](std::size_t band) {
                     const std::size_t i0 {band * transpose_tile};
                     transpose_walk(i0, std::min(i0 + transpose_tile, rows), 0, columns, piece, [&](std::size_t i, std::size_t j) {
                                      to[(j * rows) + i] = from[(i * apart) + j];
                                    });
                   });
    }

    /*!
     * \brief The transforms of \p plan's length along each of the \p columns columns of \p data into those of \p to
     *        (\p data's rows of \p columns, \p to's \p to_rows apart, as many as the plan's points): registers of neighbouring columns where they
     *        fill one (\ref fft_lane_groups), the rest up to \ref fft_column_block of them at a time, each row read and
     *        written as that many neighbours; the threads share them.
     */
    template <std::floating_point R>
    void columns_fft(const fft_plan<R>&     plan,
                     const std::complex<R>* data,
                     std::size_t            columns,
                     std::complex<R>*       to,
                     std::size_t            to_rows,
                     bool                   fused)
    {
      using C = std::complex<R>;
      const std::size_t points   {plan.length};
      const std::size_t groups   {fft_lane_groups(plan, fused, columns, false)};
      const std::size_t together {groups * fft_lines_width<R>};
      const std::size_t blocks   {(columns - together + fft_column_block - 1) / fft_column_block};
      const std::size_t work     {2 * points * static_cast<std::size_t>(std::bit_width(points))};
      parallel_for(groups + blocks, work * std::max(fft_column_block, fft_lines_width<R>), [&](std::size_t task) {
                     if (task < groups) {
                       const std::size_t column {task * fft_lines_width<R>};
                       fft_lines_together(plan, data + column, 1, columns, points, to + column, 1, to_rows, false, R(1));
                       return;
                     }
                     const std::size_t first {together + ((task - groups) * fft_column_block)};
                     const std::size_t width {std::min(fft_column_block, columns - first)};
                     fft_scratch<C, 3> scratch {plan.work + (width * points)};
                     C* const          lines {scratch.data() + plan.work};
                     for (std::size_t k {0}; k < points; ++k) {
                       for (std::size_t b {0}; b < width; ++b) {
                         lines[(b * points) + k] = data[(k * columns) + first + b];
                       }
                     }
                     for (std::size_t b {0}; b < width; ++b) {
                       forward_fft(plan, lines + (b * points), scratch.data(), fused);
                     }
                     for (std::size_t k {0}; k < points; ++k) {
                       for (std::size_t b {0}; b < width; ++b) {
                         to[(k * to_rows) + first + b] = lines[(b * points) + k];
                       }
                     }
                   });
    }

    /*!
     * \brief Each of the \p rows rows of \p data (\p apart from one another) times its run of \p twiddles (each as
     *        long as the rows), then transformed in place (\p plan's length), the row still in the caches, a row a task:
     *        rows at least a four-step transform's side, whose passes along a line alone outrun registers of them.
     */
    template <std::floating_point R>
    void twiddled_rows_fft(const fft_plan<R>&     plan,
                           std::complex<R>*       data,
                           std::size_t            rows,
                           std::size_t            apart,
                           const std::complex<R>* twiddles,
                           bool                   fused)
    {
      const std::size_t points {plan.length};
      parallel_for(rows, 2 * points * static_cast<std::size_t>(std::bit_width(points)), [&](std::size_t row) {
                     twiddle_run(data + (row * apart), twiddles + (row * points), points, fused);
                     const fft_scratch<std::complex<R>, 3> space {plan.work};
                     forward_fft(plan, data + (row * apart), space.data(), fused);
                   });
    }

    template <std::floating_point R>
    void four_step_fft(const fft_plan<R>& plan,
                       std::complex<R>*   data,
                       std::complex<R>*   work,
                       bool               fused)
    {
      const four_step_plan<R>& f  {*plan.four_step};
      const std::size_t        n1 {f.columns};
      const std::size_t        n2 {f.rows};
      const std::size_t        ld {n1 + fft_four_step_pad};
      columns_fft(*f.column_plan, data, n1, work, ld, fused);                 // work[j1 + ld·k2]
      twiddled_rows_fft(*f.row_plan, work, n2, ld, f.twiddles.data(), fused); // work[k1 + ld·k2], times w^{j1·k2} first
      four_step_transpose(work, n2, n1, ld, data);                            // data[k2 + n2·k1]
    }

    //! \brief The elements a transform reads where they lie, row-major, of shape `shape`: a tensor's, or an array's
    //!        a caller lends (the Python binding's NumPy arrays).
    template <typename T>
    struct fft_source
    {
      const T*                 data;  //!< The first element.
      std::vector<std::size_t> shape; //!< The extent of each axis.
    };

    //! \brief \p a's elements, where they lie.
    template <typename T>
    [[nodiscard]] fft_source<T> source_of(const tensor<T>& a)
    {
      return {a.data().data(), a.shape()};
    }

    /*!
     * \brief The transform of \p a along \p axis, \p n points (the axis cut or padded with zeros to it, as NumPy's),
     *        forward or \p inverse (`conj(fft(conj x))`, exact), scaled for \p norm. Each line gathered, transformed
     *        and scattered on its own, the lines shared among the threads: a line's bits are its own.
     */
    template <fft_input T>
    [[nodiscard]] tensor<typename fft_complex<T>::type> transform_axis(const fft_source<T>&       a,
                                                                       std::optional<std::size_t> n,
                                                                       std::ptrdiff_t             axis,
                                                                       fft_norm                   norm,
                                                                       bool                       inverse)
    {
      using C = typename fft_complex<T>::type;
      using R = typename C::value_type;
      const auto rank {static_cast<std::ptrdiff_t>(a.shape.size())};
      if (axis < -rank || axis >= rank) {
        throw dimension_error("fft: axis " + std::to_string(axis) + " is out of bounds for a tensor of rank " + std::to_string(rank));
      }
      const auto        index  {static_cast<std::size_t>(axis < 0 ? axis + rank : axis)};
      const std::size_t extent {a.shape[index]};
      const std::size_t points {n.value_or(extent)};
      if (points == 0) {
        throw dimension_error("fft: invalid number of data points (0)");
      }
      std::vector<std::size_t> shape {a.shape};
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
      const T* const                           from  {a.data};
      const auto                               finish {[&](const C& value) { // conjugated back for the inverse, scaled
                                                         const C v {inverse ? std::conj(value) : value};
                                                         return scale == R(1) ? v : C {v.real() * scale, v.imag() * scale};
                                                       }};
      // A line's work as work_per_thread counts it, about a nanosecond a unit: 2·n·log₂ n (a line of 64 points 0.2 µs,
      // of 1024 4.6 µs, on one arm64 core), so that 64 lines of 64 points stay on one thread.
      const std::size_t work  {2 * points * static_cast<std::size_t>(std::bit_width(points))};
      const bool        ahead {fft_starts_in_work(*plan)};
      if (walk.inner == 1) { // each line together: transformed where it lies in the output
        const std::size_t groups {fft_lane_groups(*plan, fused, walk.outer, true)};
        const std::size_t first  {groups * fft_lines_width<R>};
        parallel_for(groups + walk.outer - first, work * fft_lines_width<R>, [&](std::size_t task) {
                       if (task < groups) {
                         fft_lines_together(*plan, from + (task * fft_lines_width<R> * extent), extent, 1, kept,
                                            out.data() + (task * fft_lines_width<R> * points), points, 1, inverse, scale);
                         return;
                       }
                       const std::size_t line {first + task - groups};
                       const T* const    in   {from + (line * extent)};
                       C* const       to {out.data() + (line * points)};
                       fft_scratch<C> scratch {plan->work};
                       C* const       start {ahead ? scratch.data() : to}; // where the passes read the line
                       for (std::size_t k {0}; k < points; ++k) {
                         const C value {k < kept ? C(in[k]) : C {}};
                         start[k] = inverse ? std::conj(value) : value;
                       }
                       forward_fft(*plan, to, scratch.data(), fused, ahead);
                       if (inverse || scale != R(1)) {
                         std::transform(to, to + points, to, finish);
                       }
                     });
        return tensor<C> {std::move(shape), std::move(out)};
      }
      // Lines `inner` apart: a register of neighbours at a time where they fill one (fft_lane_groups), the rest up to
      // fft_column_block of them at a time, each row read and written as that many neighbours; fewer where that would
      // leave the threads under four tasks each. A line's bits are its own either way.
      const std::size_t groups   {fft_lane_groups(*plan, fused, walk.inner, false)};
      const std::size_t together {groups * fft_lines_width<R>};
      const std::size_t rest     {walk.inner - together};
      const std::size_t share    {(walk.outer * rest) / (4 * max_threads())};
      const std::size_t block    {std::clamp(share, std::size_t {1}, fft_column_block)};
      const std::size_t blocks   {(rest + block - 1) / block};
      parallel_for(walk.outer * (groups + blocks), work * std::max(block, fft_lines_width<R>), [&](std::size_t task) {
                     const std::size_t outer {task / (groups + blocks)};
                     const std::size_t slot  {task % (groups + blocks)};
                     if (slot < groups) {
                       const std::size_t column {slot * fft_lines_width<R>};
                       fft_lines_together(*plan, from + (outer * extent * walk.inner) + column, 1, walk.inner, kept,
                                          out.data() + (outer * points * walk.inner) + column, 1, walk.inner, inverse, scale);
                       return;
                     }
                     const std::size_t first {together + ((slot - groups) * block)};
                     const std::size_t width {std::min(block, walk.inner - first)};
                     const T* const    in    {from + (outer * extent * walk.inner) + first};
                     C* const          to    {out.data() + (outer * points * walk.inner) + first};
                     fft_scratch<C>    scratch {plan->work + (width * points)};
                     C* const          lines {scratch.data() + plan->work};
                     for (std::size_t k {0}; k < points; ++k) {
                       for (std::size_t b {0}; b < width; ++b) {
                         const C value {k < kept ? C(in[(k * walk.inner) + b]) : C {}};
                         lines[(b * points) + k] = inverse ? std::conj(value) : value;
                       }
                     }
                     for (std::size_t b {0}; b < width; ++b) {
                       forward_fft(*plan, lines + (b * points), scratch.data(), fused);
                     }
                     for (std::size_t k {0}; k < points; ++k) {
                       for (std::size_t b {0}; b < width; ++b) {
                         to[(k * walk.inner) + b] = finish(lines[(b * points) + k]);
                       }
                     }
                   });
      return tensor<C> {std::move(shape), std::move(out)};
    }

    //! \brief What a transform of `n` real points (or to them) takes: the complex plan of `n/2` points for an even `n`,
    //!        of `n` for an odd one, and for an even `n` the roots `e^{−2πi k/n}` for `k` up to `n/2`.
    template <std::floating_point R>
    struct real_fft_plan
    {
      std::size_t                        length {0}; //!< n.
      std::shared_ptr<const fft_plan<R>> complex;    //!< The complex plan: n/2 points (even n), n (odd).
      std::vector<std::complex<R>>       roots;      //!< `e^{−2πi k/n}`, k from 0 to n/2 (even n).
    };

    //! \brief The \ref real_fft_plan of \p n, shared as \ref shared_fft_plan shares the complex ones.
    template <std::floating_point R>
    [[nodiscard]] std::shared_ptr<const real_fft_plan<R>> shared_real_fft_plan(std::size_t n)
    {
      struct entry
      {
        std::size_t                             length; //!< The plan's length.
        std::shared_ptr<const real_fft_plan<R>> plan;   //!< The plan.
      };
      static std::mutex         guard;
      static std::vector<entry> plans; // the most recently used last
      {
        const std::scoped_lock lock  {guard};
        const auto             found {std::find_if(plans.begin(), plans.end(), [n](const entry& e) { return e.length == n; })};
        if (found != plans.end()) {
          std::rotate(found, found + 1, plans.end());
          return plans.back().plan;
        }
      }
      auto plan {std::make_shared<real_fft_plan<R>>()};
      plan->length = n;
      if (n % 2 == 0) {
        plan->complex = shared_fft_plan<R>(n / 2);
        const unit_roots roots {n};
        for (std::size_t k {0}; k <= n / 2; ++k) {
          const std::complex<double> root {roots.root(k)};
          plan->roots.emplace_back(static_cast<R>(root.real()), static_cast<R>(root.imag()));
        }
      }
      else {
        plan->complex = shared_fft_plan<R>(n);
      }
      const std::scoped_lock lock {guard};
      if (plans.size() >= fft_plans_kept) {
        plans.erase(plans.begin());
      }
      plans.push_back({n, plan});
      return plans.back().plan;
    }

    //! \brief No recombination by registers: the scalar loop's from `k` = 1.
    template <std::floating_point R>
    [[nodiscard]] std::size_t no_registers(std::size_t /*h*/,
                                           const std::complex<R>* /*from*/,
                                           const std::complex<R>* /*roots*/,
                                           std::complex<R>* /*into*/) noexcept
    {
      return 1;
    }

    /*!
     * \brief The first `n/2 + 1` frequencies of the \p n real points \p x into \p X by the complex transform of `n`
     *        points, the products fused where \p fused: an even `n`'s line holding an infinity or a NaN, which the
     *        recombination of its pairs would turn into NaN (`∞ − ∞`) where the complex transform carries it.
     */
    template <std::floating_point R>
    NUMERICAL_OUT_OF_LINE void real_forward_whole(std::size_t            n,
                                                  const R*               x,
                                                  std::complex<R>*       X,
                                                  bool                   fused)
    {
      using C = std::complex<R>;
      const std::shared_ptr<const fft_plan<R>> plan {shared_fft_plan<R>(n)};
      std::vector<C>                           line(n + plan->work);
      for (std::size_t j {0}; j < n; ++j) {
        line[j] = C {x[j], R(0)};
      }
      forward_fft(*plan, line.data(), line.data() + n, fused);
      std::copy(line.begin(), line.begin() + static_cast<std::ptrdiff_t>((n / 2) + 1), X);
    }

    /*!
     * \brief The \p n real points \p x from the `n/2 + 1` frequencies \p X by the complex transform of `n` points
     *        (`conj(fft(conj ·))` of the Hermitian spectrum, the imaginary parts of `X_0` and `X_{n/2}` read as 0), times
     *        \p factor: an even `n`'s spectrum holding an infinity or a NaN (\ref real_forward_whole).
     */
    template <std::floating_point R>
    NUMERICAL_OUT_OF_LINE void real_inverse_whole(std::size_t            n,
                                                  const std::complex<R>* X,
                                                  R*                     x,
                                                  R                      factor,
                                                  bool                   fused)
    {
      using C = std::complex<R>;
      const std::shared_ptr<const fft_plan<R>> plan {shared_fft_plan<R>(n)};
      const std::size_t                        h    {n / 2};
      std::vector<C>                           line(n + plan->work);
      line[0] = C {X[0].real(), R(0)};
      line[h] = C {X[h].real(), R(0)};
      for (std::size_t k {1}; k < h; ++k) {
        line[k]     = std::conj(X[k]); // conjugated for conj(fft(conj ·))
        line[n - k] = X[k];
      }
      forward_fft(*plan, line.data(), line.data() + n, fused);
      for (std::size_t j {0}; j < n; ++j) {
        x[j] = line[j].real() * factor;
      }
    }

    /*!
     * \brief The transform of \p n real points \p x (together) into \p X (`n/2 + 1` complex numbers), \p work holding
     *        `n/2 + 1` complex numbers beside the plan's work. An even `n = 2h`: the `h` complex numbers
     *        `x_{2j} + i·x_{2j+1}` transformed, then `X_k = E_k + w^k·O_k`, `E_k = (Z_k + conj Z_{h−k})/2`,
     *        `O_k = −i·(Z_k − conj Z_{h−k})/2` (exact halvings and products by ±i; `w^k·O_k` as \p Fused), \p registers
     *        `(h, Z, roots, X)` computing the first of them and returning the next `k`. An odd `n`: the complex
     *        transform of the points, its first `n/2 + 1` outputs. Inlined into each caller.
     */
    template <bool Fused,
              std::floating_point R,
              typename Registers>
    NUMERICAL_ALWAYS_INLINE void real_forward_body(const real_fft_plan<R>& plan,
                                                   const R*                x,
                                                   std::complex<R>*        X,
                                                   std::complex<R>*        work,
                                                   Registers               registers)
    {
      using C = std::complex<R>;
      const std::size_t n     {plan.length};
      const std::size_t h     {n / 2};
      const bool        ahead {fft_starts_in_work(*plan.complex)}; // the input placed where the passes end in their data
      if (n % 2 == 1) {
        C* const line  {work};
        C* const start {ahead ? work + n : line};
        for (std::size_t j {0}; j < n; ++j) {
          start[j] = C {x[j], R(0)};
        }
        forward_fft(*plan.complex, line, work + n, Fused, ahead);
        std::copy(line, line + h + 1, X);
        return;
      }
      C* const z      {work};
      C* const start  {ahead ? work + h : z};
      bool     finite {true};
      for (std::size_t j {0}; j < h; ++j) {
        start[j] = C {x[2 * j], x[(2 * j) + 1]};
        finite   = finite && std::isfinite(x[2 * j]) && std::isfinite(x[(2 * j) + 1]);
      }
      if (!finite) {
        real_forward_whole(n, x, X, Fused);
        return;
      }
      forward_fft(*plan.complex, z, work + h, Fused, ahead);
      X[0] = C {z[0].real() + z[0].imag(), R(0)};
      X[h] = C {z[0].real() - z[0].imag(), R(0)};
      for (std::size_t k {registers(h, z, plan.roots.data(), X)}; k < h; ++k) {
        const C a          {z[k]};
        const C b          {std::conj(z[h - k])};
        const C sum        {a + b};
        const C difference {a - b};
        const C even       {sum.real() * R(0.5), sum.imag() * R(0.5)};
        const C odd        {difference.imag() * R(0.5), -difference.real() * R(0.5)}; // −i·d/2
        X[k] = even + twiddled<Fused>(plan.roots[k], odd);
      }
    }

    /*!
     * \brief The inverse of \ref real_forward_body, unscaled save its own halvings: \p n real points \p x from the
     *        `n/2 + 1` complex numbers \p X (the imaginary parts of `X_0` and, for an even `n`, `X_{n/2}` read as 0,
     *        as NumPy's). An even `n = 2h`: `Z_k = E_k + i·O_k`, `E_k = (X_k + conj X_{h−k})/2`,
     *        `O_k = conj(w^k)·(X_k − conj X_{h−k})/2`, then `Z`'s inverse transform (`conj(fft(conj ·))`), its parts
     *        `x_{2j}` and `x_{2j+1}`, times \p scale, \p registers `(h, X, roots, conj Z)` computing the first `conj Z_k` and
     *        returning the next `k`; an odd `n`: the Hermitian spectrum of `n` points, its inverse transform's real parts
     *        times \p scale. Inlined into each caller.
     */
    template <bool Fused,
              std::floating_point R,
              typename Registers>
    NUMERICAL_ALWAYS_INLINE void real_inverse_body(const real_fft_plan<R>& plan,
                                                   const std::complex<R>*  X,
                                                   R*                      x,
                                                   std::complex<R>*        work,
                                                   R                       scale,
                                                   Registers               registers)
    {
      using C = std::complex<R>;
      const std::size_t n     {plan.length};
      const std::size_t h     {n / 2};
      const bool        ahead {fft_starts_in_work(*plan.complex)}; // the input placed where the passes end in their data
      if (n % 2 == 1) {
        C* const line  {work};
        C* const start {ahead ? work + n : line};
        start[0] = C {X[0].real(), R(0)};
        for (std::size_t k {1}; k <= h; ++k) {
          start[k]     = std::conj(X[k]); // conjugated for conj(fft(conj ·))
          start[n - k] = X[k];
        }
        forward_fft(*plan.complex, line, work + n, Fused, ahead);
        for (std::size_t j {0}; j < n; ++j) {
          x[j] = line[j].real() * scale;
        }
        return;
      }
      bool finite {std::isfinite(X[0].real()) && std::isfinite(X[h].real())};
      for (std::size_t k {1}; k < h; ++k) {
        finite = finite && std::isfinite(X[k].real()) && std::isfinite(X[k].imag());
      }
      if (!finite) {
        real_inverse_whole(n, X, x, scale * R(0.5), Fused); // scale holds the half sums' 2, exact
        return;
      }
      C* const z     {work};
      C* const start {ahead ? work + h : z};
      const C  first {X[0].real(), R(0)};
      const C  last  {X[h].real(), R(0)};
      start[0] = std::conj(C {(first.real() + last.real()) * R(0.5), (first.real() - last.real()) * R(0.5)});
      for (std::size_t k {registers(h, X, plan.roots.data(), start)}; k < h; ++k) {
        const C a          {X[k]};
        const C b          {std::conj(X[h - k])};
        const C sum        {a + b};
        const C difference {a - b};
        const C even       {sum.real() * R(0.5), sum.imag() * R(0.5)};
        const C odd        {twiddled<Fused>(std::conj(plan.roots[k]), C {difference.real() * R(0.5), difference.imag() * R(0.5)})};
        const C value      {even.real() - odd.imag(), even.imag() + odd.real()}; // E + i·O
        start[k] = std::conj(value);
      }
      forward_fft(*plan.complex, z, work + h, Fused, ahead);
      for (std::size_t j {0}; j < h; ++j) {
        x[2 * j]       = z[j].real() * scale;
        x[(2 * j) + 1] = -z[j].imag() * scale;
      }
    }

#ifdef NUMERICAL_X86_COMPILED
    //! \brief \ref real_forward_body fused, built for AVX2 with FMA.
    template <std::floating_point R>
    NUMERICAL_TARGET("avx2,fma") void real_forward_x86(const real_fft_plan<R>& plan,
                                                       const R*                x,
                                                       std::complex<R>*        X,
                                                       std::complex<R>*        work)
    {
      real_forward_body<true>(plan, x, X, work, avx2_real_forward_registers<R>);
    }

    //! \brief \ref real_inverse_body fused, built for AVX2 with FMA.
    template <std::floating_point R>
    NUMERICAL_TARGET("avx2,fma") void real_inverse_x86(const real_fft_plan<R>& plan,
                                                       const std::complex<R>*  X,
                                                       R*                      x,
                                                       std::complex<R>*        work,
                                                       R                       scale)
    {
      real_inverse_body<true>(plan, X, x, work, scale, avx2_real_inverse_registers<R>);
    }

#endif

    //! \brief \ref real_forward_body on the path in use.
    template <std::floating_point R>
    void real_forward(const real_fft_plan<R>& plan,
                      const R*                x,
                      std::complex<R>*        X,
                      std::complex<R>*        work,
                      bool                    fused)
    {
      if (!fused) {
        real_forward_body<false>(plan, x, X, work, no_registers<R>);
        return;
      }
#ifdef NUMERICAL_X86_COMPILED
      real_forward_x86(plan, x, X, work);
#else
      real_forward_body<true>(plan, x, X, work, no_registers<R>);
#endif
    }

    //! \brief \ref real_inverse_body on the path in use.
    template <std::floating_point R>
    void real_inverse(const real_fft_plan<R>& plan,
                      const std::complex<R>*  X,
                      R*                      x,
                      std::complex<R>*        work,
                      R                       scale,
                      bool                    fused)
    {
      if (!fused) {
        real_inverse_body<false>(plan, X, x, work, scale, no_registers<R>);
        return;
      }
#ifdef NUMERICAL_X86_COMPILED
      real_inverse_x86(plan, X, x, work, scale);
#else
      real_inverse_body<true>(plan, X, x, work, scale, no_registers<R>);
#endif
    }

    /*!
     * \brief A transform of each line of \p a along axis \p index into lines of \p out_points: each line gathered (its
     *        first \p in_points elements, padded with zeros), passed with its output line, both together, and the
     *        thread's working space of \p work elements of `C` to \p line, then scattered; a line together and not
     *        padded passed where it lies, its output line too. The lines shared among the threads.
     */
    template <typename In,
              typename Out,
              typename C,
              typename Line>
    [[nodiscard]] tensor<Out> lines_along(const fft_source<In>& a,
                                          std::size_t           index,
                                          std::size_t           in_points,
                                          std::size_t           out_points,
                                          std::size_t           work,
                                          Line                  line)
    {
      std::vector<std::size_t> shape  {a.shape};
      const std::size_t        extent {shape[index]};
      shape[index] = out_points;
      const axis_walk walk            {walk_along(shape, index)};
      buffer<Out>     out(element_count(shape.begin(), shape.end()));
      if (out.empty()) {
        return tensor<Out> {std::move(shape), std::move(out)};
      }
      const std::size_t kept {std::min(in_points, extent)};
      const In* const   from {a.data};
      if (walk.inner == 1 && in_points <= extent) { // each line together, none padded: read and written where it lies
        parallel_for(walk.outer, 5 * (in_points + out_points) * static_cast<std::size_t>(std::bit_width(in_points + out_points)),
                     [&](std::size_t number) {
                       fft_scratch<C, 0> space {work};
                       line(from + (number * extent), out.data() + (number * out_points), space.data());
                     });
        return tensor<Out> {std::move(shape), std::move(out)};
      }
      parallel_for(walk.outer * walk.inner, 5 * (in_points + out_points) * static_cast<std::size_t>(std::bit_width(in_points + out_points)),
                   [&](std::size_t number) {
                     const std::size_t      outer {number / walk.inner};
                     const std::size_t      inner {number % walk.inner};
                     const In* const        in    {from + (outer * extent * walk.inner) + inner};
                     Out* const             to    {out.data() + (outer * out_points * walk.inner) + inner};
                     fft_scratch<C, 0>      space {work};
                     fft_scratch<In, 1>     gathered {in_points};
                     fft_scratch<Out, 2>    result {out_points};
                     for (std::size_t k {0}; k < in_points; ++k) {
                       gathered.data()[k] = k < kept ? in[k * walk.inner] : In {};
                     }
                     line(gathered.data(), result.data(), space.data());
                     for (std::size_t k {0}; k < out_points; ++k) {
                       to[k * walk.inner] = result.data()[k];
                     }
                   });
      return tensor<Out> {std::move(shape), std::move(out)};
    }

    //! \brief \p axis of a tensor of rank \p rank from 0, NumPy's negative axes read from the end.
    //!        \throws dimension_error If \p axis is not an axis.
    [[nodiscard]] inline std::size_t fft_axis_index(std::ptrdiff_t axis,
                                                    std::size_t    rank,
                                                    const char*    who)
    {
      const auto signed_rank {static_cast<std::ptrdiff_t>(rank)};
      if (axis < -signed_rank || axis >= signed_rank) {
        throw dimension_error(std::string {who} + ": axis " + std::to_string(axis) + " is out of bounds for a tensor of rank " + std::to_string(rank));
      }
      return static_cast<std::size_t>(axis < 0 ? axis + signed_rank : axis);
    }

    /*!
     * \brief The transform of real \p a along the axis \p index into all `n` frequencies, or the inverse transform where
     *        \p inverse: each line's real transform (`real_forward`, its `n/2 + 1` non-negative frequencies), the rest
     *        by Hermitian symmetry, `X_{n−k} = conj X_k`, exactly; a length whose complex transform takes four steps,
     *        which half of it does not, by that complex transform, on the threads. The inverse of a real line the
     *        conjugate of its transform. Then times \p scale.
     */
    template <std::floating_point R>
    [[nodiscard]] tensor<std::complex<R>> real_input_transform(const fft_source<R>& a,
                                                               std::size_t          index,
                                                               std::size_t          points,
                                                               bool                 inverse,
                                                               R                    scale)
    {
      using C = std::complex<R>;
      const bool                                    whole   {four_step_columns(points, sizeof(C)) != 0};
      const std::shared_ptr<const real_fft_plan<R>> plan    {whole ? nullptr : shared_real_fft_plan<R>(points)};
      const std::shared_ptr<const fft_plan<R>>      complex {whole ? shared_fft_plan<R>(points) : nullptr};
      const bool                                    fused   {fft_fuses()};
      const std::size_t                             half    {points / 2};
      const std::size_t                             work    {whole ? complex->work : (points % 2 == 0 ? half : points) + plan->complex->work};
      return lines_along<R, C, C>(a, index, points, points, work, [&](const R* x, C* spectrum, C* space) {
                                    if (whole) {
                                      std::transform(x, x + points, spectrum, [](R value) { return C {value, R(0)}; });
                                      forward_fft(*complex, spectrum, space, fused);
                                    }
                                    else {
                                      real_forward(*plan, x, spectrum, space, fused);
                                      for (std::size_t k {1}; k < points - half; ++k) {
                                        spectrum[points - k] = std::conj(spectrum[k]);
                                      }
                                    }
                                    if (inverse || scale != R(1)) {
                                      std::transform(spectrum, spectrum + points, spectrum, [inverse, scale](const C& v) {
                                                       const C w {inverse ? std::conj(v) : v};
                                                       return C {w.real() * scale, w.imag() * scale};
                                                     });
                                    }
                                  });
    }

    /*!
     * \brief `fft` or `ifft` (\p inverse) of the real elements \p a: `fft` of a real tensor's computation, of
     *        elements a caller lends.
     */
    template <std::floating_point T>
    [[nodiscard]] tensor<std::complex<T>> real_transform_of(const fft_source<T>&       a,
                                                            std::optional<std::size_t> n,
                                                            std::ptrdiff_t             axis,
                                                            fft_norm                   norm,
                                                            bool                       inverse)
    {
      const char* const name   {inverse ? "ifft" : "fft"};
      const std::size_t index  {fft_axis_index(axis, a.shape.size(), name)};
      const std::size_t points {n.value_or(a.shape[index])};
      if (points == 0) {
        throw dimension_error(std::string {name} + ": invalid number of data points (0)");
      }
      return real_input_transform(a, index, points, inverse, fft_scale<T>(norm, inverse, points));
    }

    //! \brief `rfft`'s computation, of real elements a caller lends.
    template <std::floating_point T>
    [[nodiscard]] tensor<std::complex<T>> rfft_of(const fft_source<T>&       a,
                                                  std::optional<std::size_t> n,
                                                  std::ptrdiff_t             axis,
                                                  fft_norm                   norm)
    {
      using C = std::complex<T>;
      const std::size_t index  {fft_axis_index(axis, a.shape.size(), "rfft")};
      const std::size_t points {n.value_or(a.shape[index])};
      if (points == 0) {
        throw dimension_error("rfft: invalid number of data points (0)");
      }
      const std::shared_ptr<const real_fft_plan<T>>          plan  {shared_real_fft_plan<T>(points)};
      const bool                                             fused {fft_fuses()};
      const T                                                scale {fft_scale<T>(norm, false, points)};
      const std::size_t                                      half  {points / 2};
      const std::size_t                                      work  {(points % 2 == 0 ? half : points) + plan->complex->work};
      return lines_along<T, C, C>(a, index, points, half + 1, work, [&](const T* x, C* spectrum, C* space) {
                                    real_forward(*plan, x, spectrum, space, fused);
                                    if (scale != T(1)) {
                                      std::transform(spectrum, spectrum + half + 1, spectrum, [scale](const C& v) { return C {v.real() * scale, v.imag() * scale}; });
                                    }
                                  });
    }

    //! \brief `irfft`'s computation, of complex elements a caller lends.
    template <std::floating_point T>
    [[nodiscard]] tensor<T> irfft_of(const fft_source<std::complex<T>>& a,
                                     std::optional<std::size_t>         n,
                                     std::ptrdiff_t                     axis,
                                     fft_norm                           norm)
    {
      using C = std::complex<T>;
      const std::size_t index  {fft_axis_index(axis, a.shape.size(), "irfft")};
      const std::size_t extent {a.shape[index]};
      const std::size_t points {n.value_or(extent == 0 ? 0 : 2 * (extent - 1))};
      if (points == 0) {
        throw dimension_error("irfft: invalid number of data points (0)");
      }
      const std::shared_ptr<const real_fft_plan<T>>          plan   {shared_real_fft_plan<T>(points)};
      const bool                                             fused  {fft_fuses()};
      const T                                                factor {fft_scale<T>(norm, true, points)};
      const T                                                scale  {points % 2 == 0 ? factor * T(2) : factor}; // an even n's half sums
      const std::size_t                                      half   {points / 2};
      const std::size_t                                      work   {(points % 2 == 0 ? half : points) + plan->complex->work};
      return lines_along<C, T, C>(a, index, half + 1, points, work, [&](const C* spectrum, T* x, C* space) {
                                    real_inverse(*plan, spectrum, x, space, scale, fused);
                                  });
    }
  } // namespace detail

  /*!
   * \brief The discrete Fourier transform of \p a along \p axis, as `numpy.fft.fft`: `X_k = Σ_j x_j e^{−2πi jk/n}`.
   * \param[in] a    A tensor of complex `float` or `double` (a real one: the overload of real tensors).
   * \param[in] n    The points transformed: the axis cut to them, or padded with zeros; its extent by default.
   * \param[in] axis The axis transformed, negative from the last (the last by default).
   * \param[in] norm The scaling (\ref fft_norm): none by default.
   * \return The complex transform, of `a`'s shape with the axis `n` long.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <detail::fft_input T>
  requires detail::split_complex<T>
  [[nodiscard]] tensor<T> fft(const tensor<T>&           a,
                              std::optional<std::size_t> n    = std::nullopt,
                              std::ptrdiff_t             axis = -1,
                              fft_norm                   norm = fft_norm::backward)
  {
    return detail::transform_axis(detail::source_of(a), n, axis, norm, false);
  }

  /*!
   * \brief The discrete Fourier transform of real \p a along \p axis, as `numpy.fft.fft` of a real array: its
   *        non-negative frequencies by the real transform (`rfft`'s), the others by their Hermitian symmetry,
   *        `X_{n−k} = conj X_k` exactly; a length whose complex transform takes four steps (`four_step_plan`) as a
   *        complex one, on the threads.
   * \param[in] a    A tensor of `float` or `double`.
   * \param[in] n    The points transformed: the axis cut to them, or padded with zeros; its extent by default.
   * \param[in] axis The axis transformed, negative from the last (the last by default).
   * \param[in] norm The scaling (\ref fft_norm): none by default.
   * \return The complex transform, of `a`'s shape with the axis `n` long.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <std::floating_point T>
  [[nodiscard]] tensor<std::complex<T>> fft(const tensor<T>&           a,
                                            std::optional<std::size_t> n    = std::nullopt,
                                            std::ptrdiff_t             axis = -1,
                                            fft_norm                   norm = fft_norm::backward)
  {
    return detail::real_transform_of(detail::source_of(a), n, axis, norm, false);
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
  requires detail::split_complex<T>
  [[nodiscard]] tensor<T> ifft(const tensor<T>&           a,
                               std::optional<std::size_t> n    = std::nullopt,
                               std::ptrdiff_t             axis = -1,
                               fft_norm                   norm = fft_norm::backward)
  {
    return detail::transform_axis(detail::source_of(a), n, axis, norm, true);
  }

  /*!
   * \brief The inverse discrete Fourier transform of real \p a along \p axis, as `numpy.fft.ifft` of a real array: the
   *        conjugate of its transform (`fft` of a real tensor), scaled as the inverse.
   * \param[in] a    A tensor of `float` or `double`.
   * \param[in] n    The points transformed: the axis cut to them, or padded with zeros; its extent by default.
   * \param[in] axis The axis transformed, negative from the last (the last by default).
   * \param[in] norm The scaling (\ref fft_norm): 1/n by default.
   * \return The complex inverse transform, of `a`'s shape with the axis `n` long.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <std::floating_point T>
  [[nodiscard]] tensor<std::complex<T>> ifft(const tensor<T>&           a,
                                             std::optional<std::size_t> n    = std::nullopt,
                                             std::ptrdiff_t             axis = -1,
                                             fft_norm                   norm = fft_norm::backward)
  {
    return detail::real_transform_of(detail::source_of(a), n, axis, norm, true);
  }

  /*!
   * \brief \p a with its axis \p axis rolled so that its element 0 lands at its middle, as `numpy.fft.fftshift` along
   *        one axis: element `j` to `(j + n/2) mod n`; or back, as `numpy.fft.ifftshift`, where \p inverse.
   * \param[in] a       Any tensor.
   * \param[in] axis    The axis rolled, negative from the last.
   * \param[in] inverse `ifftshift`'s roll, by `−(n/2)`, where an odd extent makes it differ.
   * \return The rolled copy, of `a`'s shape.
   * \throws dimension_error If \p axis is not an axis of \p a.
   */
  template <typename T>
  [[nodiscard]] tensor<T> fftshift(const tensor<T>& a,
                                   std::ptrdiff_t   axis    = -1,
                                   bool             inverse = false)
  {
    const auto rank {static_cast<std::ptrdiff_t>(a.rank())};
    if (axis < -rank || axis >= rank) {
      throw dimension_error("fftshift: axis " + std::to_string(axis) + " is out of bounds for a tensor of rank " + std::to_string(rank));
    }
    const auto              index {static_cast<std::size_t>(axis < 0 ? axis + rank : axis)};
    const detail::axis_walk walk  {detail::walk_along(a.shape(), index)};
    const std::size_t       n     {walk.extent};
    buffer<T>               out(a.data().size());
    if (out.empty()) { // an empty axis, or an empty one besides it
      return tensor<T> {a.shape(), std::move(out)};
    }
    const std::size_t shift {inverse ? n - (n / 2) : n / 2}; // element j goes to (j + shift) mod n
    for (std::size_t outer {0}; outer < walk.outer; ++outer) {
      const T* const from {a.data().data() + (outer * n * walk.inner)};
      T* const       to   {out.data() + (outer * n * walk.inner)};
      for (std::size_t j {0}; j < n; ++j) {
        const std::size_t target {(j + shift) % n};
        std::copy(from + (j * walk.inner), from + ((j + 1) * walk.inner), to + (target * walk.inner));
      }
    }
    return tensor<T> {a.shape(), std::move(out)};
  }

  /*!
   * \brief The discrete Fourier transform of real \p a along \p axis, its `n/2 + 1` non-negative frequencies, as
   *        `numpy.fft.rfft`. An even `n` by the complex transform of its `n/2` pairs and their recombination, an odd one
   *        by the complex transform of its points.
   * \param[in] a    A tensor of `float` or `double`.
   * \param[in] n    The points transformed: the axis cut to them, or padded with zeros; its extent by default.
   * \param[in] axis The axis transformed, negative from the last (the last by default).
   * \param[in] norm The scaling (\ref fft_norm): none by default.
   * \return The complex transform, of `a`'s shape with the axis `n/2 + 1` long.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <std::floating_point T>
  [[nodiscard]] tensor<std::complex<T>> rfft(const tensor<T>&           a,
                                             std::optional<std::size_t> n    = std::nullopt,
                                             std::ptrdiff_t             axis = -1,
                                             fft_norm                   norm = fft_norm::backward)
  {
    return detail::rfft_of(detail::source_of(a), n, axis, norm);
  }

  /*!
   * \brief The inverse of \ref rfft along \p axis, as `numpy.fft.irfft`: `n` real points from the first `n/2 + 1`
   *        frequencies of \p a (cut, or padded with zeros), the imaginary parts of the zero frequency and of an even
   *        `n`'s last read as 0.
   * \param[in] a    A tensor of complex `float` or `double`.
   * \param[in] n    The real points: `2·(m − 1)` by default, `m` the axis' extent.
   * \param[in] axis The axis transformed, negative from the last (the last by default).
   * \param[in] norm The scaling (\ref fft_norm): 1/n by default.
   * \return The real inverse transform, of `a`'s shape with the axis `n` long.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <std::floating_point T>
  [[nodiscard]] tensor<T> irfft(const tensor<std::complex<T>>& a,
                                std::optional<std::size_t>     n    = std::nullopt,
                                std::ptrdiff_t                 axis = -1,
                                fft_norm                       norm = fft_norm::backward)
  {
    return detail::irfft_of(detail::source_of(a), n, axis, norm);
  }

  //! \brief \p norm with its forward and backward exchanged, as NumPy's `hfft` and `ihfft` take it.
  [[nodiscard]] constexpr fft_norm swapped(fft_norm norm) noexcept
  {
    return norm == fft_norm::backward ? fft_norm::forward : norm == fft_norm::forward ? fft_norm::backward : norm;
  }

  /*!
   * \brief The transform of a signal of Hermitian symmetry, its `n` real points from the non-negative frequencies \p a,
   *        as `numpy.fft.hfft`: \ref irfft of `conj(a)`, its scaling's direction exchanged.
   * \param[in] a    A tensor of complex `float` or `double`.
   * \param[in] n    The real points: `2·(m − 1)` by default.
   * \param[in] axis The axis transformed, negative from the last.
   * \param[in] norm The scaling (\ref fft_norm): none by default.
   * \return The real transform.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <std::floating_point T>
  [[nodiscard]] tensor<T> hfft(const tensor<std::complex<T>>& a,
                               std::optional<std::size_t>     n    = std::nullopt,
                               std::ptrdiff_t                 axis = -1,
                               fft_norm                       norm = fft_norm::backward)
  {
    buffer<std::complex<T>> conjugated(a.data().size());
    std::transform(a.data().begin(), a.data().end(), conjugated.begin(), [](const std::complex<T>& v) { return std::conj(v); });
    return irfft(tensor<std::complex<T>> {a.shape(), std::move(conjugated)}, n, axis, swapped(norm));
  }

  /*!
   * \brief The inverse of \ref hfft, as `numpy.fft.ihfft`: `conj(rfft(a))`, its scaling's direction exchanged.
   * \param[in] a    A tensor of `float` or `double`.
   * \param[in] n    The points transformed: the axis' extent by default.
   * \param[in] axis The axis transformed, negative from the last.
   * \param[in] norm The scaling (\ref fft_norm): 1/n by default.
   * \return The `n/2 + 1` complex frequencies.
   * \throws dimension_error If \p axis is not an axis of \p a, or \p n is 0.
   */
  template <std::floating_point T>
  [[nodiscard]] tensor<std::complex<T>> ihfft(const tensor<T>&           a,
                                              std::optional<std::size_t> n    = std::nullopt,
                                              std::ptrdiff_t             axis = -1,
                                              fft_norm                   norm = fft_norm::backward)
  {
    tensor<std::complex<T>> spectrum {rfft(a, n, axis, swapped(norm))};
    buffer<std::complex<T>> conjugated(spectrum.data().size());
    std::transform(spectrum.data().begin(), spectrum.data().end(), conjugated.begin(), [](const std::complex<T>& v) { return std::conj(v); });
    return tensor<std::complex<T>> {spectrum.shape(), std::move(conjugated)};
  }
} // namespace numerical

#endif // NUMERICAL_FFT_HPP
