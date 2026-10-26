/*!
 * \file fft_kernels.hpp
 * \brief The transform's radix-4 and radix-16 passes in registers of complex numbers, written once for every unit
 *        (AVX2, AVX-512, NEON): each operation overloaded on a register of the unit (x86.hpp's on x86-64, NEON's here)
 *        and the passes' bodies inlined into a function of each, so that every unit runs the same operations lane by
 *        lane, whatever its width: the same bits.
 */
#ifndef NUMERICAL_FFT_KERNELS_HPP
#define NUMERICAL_FFT_KERNELS_HPP

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

#include "gemm.hpp"
#include "simd.hpp"
#include "tensor.hpp"
#include "x86.hpp"

namespace numerical::detail {

#if defined(NUMERICAL_NEON_COMPILED)
  //! \brief NEON's registers of the transform's kernels.
  struct fft_neon
  {
    static constexpr bool whole_strides {false}; //!< A stride of `float` may leave a register half filled.
  };

  //! \brief A register of complex numbers of \p R that \p Isa holds, as the kernels pass it (x86.hpp's on x86-64).
  template <typename Isa,
            typename R>
  struct fft_reg;

  //! \brief One complex `double` (NEON).
  template <>
  struct fft_reg<fft_neon, double>
  {
    float64x2_t v; //!< The register.
  };

  //! \brief Two complex `float` (NEON).
  template <>
  struct fft_reg<fft_neon, float>
  {
    float32x4_t v; //!< The register.
  };

  //! \brief The sign bit of each real part's lanes (\p real) or each imaginary part's, as `double`'s lanes.
  NUMERICAL_ALWAYS_INLINE uint64x2_t neon_part_signs(fft_reg<fft_neon, double> /*kind*/,
                                                     bool real) noexcept
  {
    const std::uint64_t sign {std::uint64_t {1} << 63U};
    return real ? vcombine_u64(vcreate_u64(sign), vcreate_u64(0)) : vcombine_u64(vcreate_u64(0), vcreate_u64(sign));
  }

  //! \brief \ref neon_part_signs of `float`: each 64-bit pair a complex number, its real part the low half.
  NUMERICAL_ALWAYS_INLINE uint64x2_t neon_part_signs(fft_reg<fft_neon, float> /*kind*/,
                                                     bool real) noexcept
  {
    return vdupq_n_u64(std::uint64_t {1} << (real ? 31U : 63U));
  }

  //! \brief \p v with the signs \p signs marks flipped, exact.
  NUMERICAL_ALWAYS_INLINE float64x2_t neon_flipped(float64x2_t v,
                                                   uint64x2_t  signs) noexcept
  {
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v), signs));
  }

  //! \brief \ref neon_flipped of `float`.
  NUMERICAL_ALWAYS_INLINE float32x4_t neon_flipped(float32x4_t v,
                                                   uint64x2_t  signs) noexcept
  {
    return vreinterpretq_f32_u64(veorq_u64(vreinterpretq_u64_f32(v), signs));
  }

  //! \brief The register at \p p.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_load(fft_neon /*isa*/,
                                                             const double* p) noexcept
  {
    return {vld1q_f64(p)};
  }

  //! \brief \p c in every lane.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_splat(fft_neon /*isa*/,
                                                              double c) noexcept
  {
    return {vdupq_n_f64(c)};
  }

  //! \brief \p v stored at \p p.
  NUMERICAL_ALWAYS_INLINE void fft_store(double                  * p,
                                         fft_reg<fft_neon, double> v) noexcept
  {
    vst1q_f64(p, v.v);
  }

  //! \brief `a + b`, lane by lane.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_add(fft_reg<fft_neon, double> a,
                                                            fft_reg<fft_neon, double> b) noexcept
  {
    return {vaddq_f64(a.v, b.v)};
  }

  //! \brief `a − b`, lane by lane.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_sub(fft_reg<fft_neon, double> a,
                                                            fft_reg<fft_neon, double> b) noexcept
  {
    return {vsubq_f64(a.v, b.v)};
  }

  //! \brief The complex numbers of \p v times i: each `(−im, re)`, exact.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_times_i(fft_reg<fft_neon, double> v) noexcept
  {
    return {neon_flipped(neon_exchanged(v.v), neon_part_signs(v, true))};
  }

  //! \brief The complex numbers of \p v times −i: each `(im, −re)`, exact.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_times_minus_i(fft_reg<fft_neon, double> v) noexcept
  {
    return {neon_flipped(neon_exchanged(v.v), neon_part_signs(v, false))};
  }

  //! \brief `w · v` of each complex number of \p v, \p re and \p im `w`'s parts in every lane, as `fused_product`:
  //!        `w.im · v`'s parts exchanged and rounded, its real lanes negated (exact), then `w.re · v` fused with them.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_twiddled(fft_reg<fft_neon, double> re,
                                                                 fft_reg<fft_neon, double> im,
                                                                 fft_reg<fft_neon, double> v) noexcept
  {
    const float64x2_t crossed {neon_flipped(vmulq_f64(im.v, neon_exchanged(v.v)), neon_part_signs(v, true))};
    return {vfmaq_f64(crossed, re.v, v.v)};
  }

  //! \brief The register at \p p of `float`.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_load(fft_neon /*isa*/,
                                                            const float* p) noexcept
  {
    return {vld1q_f32(p)};
  }

  //! \brief \p c in every lane of `float`.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_splat(fft_neon /*isa*/,
                                                             float c) noexcept
  {
    return {vdupq_n_f32(c)};
  }

  //! \brief \p v stored at \p p of `float`.
  NUMERICAL_ALWAYS_INLINE void fft_store(float                  * p,
                                         fft_reg<fft_neon, float> v) noexcept
  {
    vst1q_f32(p, v.v);
  }

  //! \brief `a + b`, lane by lane of `float`.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_add(fft_reg<fft_neon, float> a,
                                                           fft_reg<fft_neon, float> b) noexcept
  {
    return {vaddq_f32(a.v, b.v)};
  }

  //! \brief `a − b`, lane by lane of `float`.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_sub(fft_reg<fft_neon, float> a,
                                                           fft_reg<fft_neon, float> b) noexcept
  {
    return {vsubq_f32(a.v, b.v)};
  }

  //! \brief The complex numbers of \p v times i: each `(−im, re)`, exact of `float`.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_times_i(fft_reg<fft_neon, float> v) noexcept
  {
    return {neon_flipped(neon_exchanged(v.v), neon_part_signs(v, true))};
  }

  //! \brief The complex numbers of \p v times −i: each `(im, −re)`, exact of `float`.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_times_minus_i(fft_reg<fft_neon, float> v) noexcept
  {
    return {neon_flipped(neon_exchanged(v.v), neon_part_signs(v, false))};
  }

  //! \brief `w · v` of each complex number of \p v, \p re and \p im `w`'s parts in every lane, as `fused_product`:
  //!        `w.im · v`'s parts exchanged and rounded, its real lanes negated (exact), then `w.re · v` fused with them of `float`.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_twiddled(fft_reg<fft_neon, float> re,
                                                                fft_reg<fft_neon, float> im,
                                                                fft_reg<fft_neon, float> v) noexcept
  {
    const float32x4_t crossed {neon_flipped(vmulq_f32(im.v, neon_exchanged(v.v)), neon_part_signs(v, true))};
    return {vfmaq_f32(crossed, re.v, v.v)};
  }

  //! \brief The real parts of the complex number of \p v, in both lanes.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_real_parts(fft_reg<fft_neon, double> v) noexcept
  {
    return {vdupq_laneq_f64(v.v, 0)};
  }

  //! \brief \ref fft_real_parts of `float`: each complex number's in both of its lanes.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_real_parts(fft_reg<fft_neon, float> v) noexcept
  {
    return {vtrn1q_f32(v.v, v.v)};
  }

  //! \brief The imaginary part of the complex number of \p v, in both lanes.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_imaginary_parts(fft_reg<fft_neon, double> v) noexcept
  {
    return {vdupq_laneq_f64(v.v, 1)};
  }

  //! \brief \ref fft_imaginary_parts of `float`: each complex number's in both of its lanes.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_imaginary_parts(fft_reg<fft_neon, float> v) noexcept
  {
    return {vtrn2q_f32(v.v, v.v)};
  }

  //! \brief The outputs of a register of radix-4 butterflies (one, of `double`), \p y0 to \p y3 its outputs, stored in
  //!        order at \p to.
  NUMERICAL_ALWAYS_INLINE void fft_store_butterflies(double*                    to,
                                                     fft_reg<fft_neon, double>  y0,
                                                     fft_reg<fft_neon, double>  y1,
                                                     fft_reg<fft_neon, double>  y2,
                                                     fft_reg<fft_neon, double>  y3) noexcept
  {
    vst1q_f64(to, y0.v);
    vst1q_f64(to + 2, y1.v);
    vst1q_f64(to + 4, y2.v);
    vst1q_f64(to + 6, y3.v);
  }

  //! \brief \ref fft_store_butterflies of two butterflies of complex `float`: their 64-bit complex numbers paired.
  NUMERICAL_ALWAYS_INLINE void fft_store_butterflies(float*                    to,
                                                     fft_reg<fft_neon, float>  y0,
                                                     fft_reg<fft_neon, float>  y1,
                                                     fft_reg<fft_neon, float>  y2,
                                                     fft_reg<fft_neon, float>  y3) noexcept
  {
    const float64x2_t a {vreinterpretq_f64_f32(y0.v)};
    const float64x2_t b {vreinterpretq_f64_f32(y1.v)};
    const float64x2_t c {vreinterpretq_f64_f32(y2.v)};
    const float64x2_t d {vreinterpretq_f64_f32(y3.v)};
    vst1q_f32(to, vreinterpretq_f32_f64(vzip1q_f64(a, b)));      // butterfly 0: y0 y1
    vst1q_f32(to + 4, vreinterpretq_f32_f64(vzip1q_f64(c, d)));  //              y2 y3
    vst1q_f32(to + 8, vreinterpretq_f32_f64(vzip2q_f64(a, b)));  // butterfly 1
    vst1q_f32(to + 12, vreinterpretq_f32_f64(vzip2q_f64(c, d)));
  }

  //! \brief `a · b`, lane by lane, rounded.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_mul(fft_reg<fft_neon, double> a,
                                                            fft_reg<fft_neon, double> b) noexcept
  {
    return {vmulq_f64(a.v, b.v)};
  }

  //! \brief `a · b + c`, lane by lane, one rounding.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_fused(fft_reg<fft_neon, double> a,
                                                              fft_reg<fft_neon, double> b,
                                                              fft_reg<fft_neon, double> c) noexcept
  {
    return {vfmaq_f64(c.v, a.v, b.v)};
  }

  //! \brief Each complex number's real part from \p even, its imaginary part from \p odd.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_parts_of(fft_reg<fft_neon, double> even,
                                                                 fft_reg<fft_neon, double> odd) noexcept
  {
    return {vcopyq_laneq_f64(even.v, 1, odd.v, 1)};
  }

  //! \brief The complex numbers of \p v conjugated, exact.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_conjugated(fft_reg<fft_neon, double> v) noexcept
  {
    return {neon_flipped(v.v, neon_part_signs(v, false))};
  }

  //! \brief Each complex number of \p v, its parts exchanged.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, double> fft_exchanged(fft_reg<fft_neon, double> v) noexcept
  {
    return {neon_exchanged(v.v)};
  }

  //! \brief `a · b`, lane by lane, rounded.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_mul(fft_reg<fft_neon, float> a,
                                                           fft_reg<fft_neon, float> b) noexcept
  {
    return {vmulq_f32(a.v, b.v)};
  }

  //! \brief `a · b + c`, lane by lane, one rounding.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_fused(fft_reg<fft_neon, float> a,
                                                             fft_reg<fft_neon, float> b,
                                                             fft_reg<fft_neon, float> c) noexcept
  {
    return {vfmaq_f32(c.v, a.v, b.v)};
  }

  //! \brief Each complex number's real part from \p even, its imaginary part from \p odd.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_parts_of(fft_reg<fft_neon, float> even,
                                                                fft_reg<fft_neon, float> odd) noexcept
  {
    return {vbslq_f32(vreinterpretq_u32_u64(vdupq_n_u64(std::uint64_t {0xFFFFFFFF} << 32U)), odd.v, even.v)};
  }

  //! \brief The complex numbers of \p v conjugated, exact.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_conjugated(fft_reg<fft_neon, float> v) noexcept
  {
    return {neon_flipped(v.v, neon_part_signs(v, false))};
  }

  //! \brief Each complex number of \p v, its parts exchanged.
  NUMERICAL_ALWAYS_INLINE fft_reg<fft_neon, float> fft_exchanged(fft_reg<fft_neon, float> v) noexcept
  {
    return {neon_exchanged(v.v)};
  }

#endif

#if defined(NUMERICAL_X86_COMPILED) || defined(NUMERICAL_NEON_COMPILED)

  //! \brief `s + c · v` of complex numbers and a real \p c, each part one fused multiply-add (`fft.hpp`'s fused
  //!        `scaled_add`).
  template <typename R>
  NUMERICAL_ALWAYS_INLINE std::complex<R> fft_scaled_add(const std::complex<R>& s,
                                                         R                      c,
                                                         const std::complex<R>& v) noexcept
  {
    return {std::fma(c, v.real(), s.real()), std::fma(c, v.imag(), s.imag())};
  }

  //! \brief `w · v` of complex numbers, one at a time, as \ref fft_twiddled computes each lane.
  template <typename R>
  NUMERICAL_ALWAYS_INLINE std::complex<R> fft_fused_product(const std::complex<R>& w,
                                                            const std::complex<R>& v) noexcept
  {
    const R ii {w.imag() * v.imag()};
    const R ir {w.imag() * v.real()};
    return {std::fma(w.real(), v.real(), -ii), std::fma(w.real(), v.imag(), ir)};
  }

  //! \brief `fft.hpp`'s `four_point` over registers of complex numbers: \p v0 to \p v3 into \p y0 to \p y3.
  template <typename V>
  NUMERICAL_ALWAYS_INLINE void fft_four_point(const V& v0,
                                              const V& v1,
                                              const V& v2,
                                              const V& v3,
                                              V&       y0,
                                              V&       y1,
                                              V&       y2,
                                              V&       y3) noexcept
  {
    const V a_plus_c  {fft_add(v0, v2)};
    const V a_minus_c {fft_sub(v0, v2)};
    const V b_plus_d  {fft_add(v1, v3)};
    const V i_b_d     {fft_times_i(fft_sub(v1, v3))};
    y0 = fft_add(a_plus_c, b_plus_d);
    y1 = fft_sub(a_minus_c, i_b_d);
    y2 = fft_sub(a_plus_c, b_plus_d);
    y3 = fft_add(a_minus_c, i_b_d);
  }

  /*!
   * \brief One Stockham pass of radix 4 of `fft.hpp`'s (`radix_four_pass`), its factors' products fused as
   *        `fused_product`'s: butterflies `(p, q)` a register of complex numbers of \p Isa along `q` at a time where the
   *        stride holds one, the first pass's a register along `p`, else one at a time. \p twiddles holds the pass'
   *        `w^p`, then `w^{2p}`, then `w^{3p}`, from `p` = 1. Inlined into the functions built for \p Isa.
   */
  template <typename Isa,
            typename R>
  NUMERICAL_ALWAYS_INLINE void fft_radix_four_pass(std::size_t            current,
                                                   std::size_t            stride,
                                                   const std::complex<R>* in,
                                                   std::complex<R>*       out,
                                                   const std::complex<R>* twiddles) noexcept
  {
    using C   = std::complex<R>;
    using vec = fft_reg<Isa, R>;
    constexpr Isa         isa          {};
    constexpr std::size_t width        {sizeof(vec) / sizeof(C)};        // complex numbers in a register
    const std::size_t     quarter      {current / 4};
    const std::size_t     run          {quarter - 1};
    const std::size_t     apart        {2 * stride * quarter};           // parts between a butterfly's inputs
    const R* const        from         {reinterpret_cast<const R*>(in)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
    R* const              into         {reinterpret_cast<R*>(out)};      // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::size_t           vector_begin {quarter};                        // the butterflies the first pass takes by registers: [vector_begin, vector_end)
    std::size_t           vector_end   {quarter};
    if (stride == 1 && quarter >= 2 * width) {                       // the first pass: a register of butterflies along p, past the first
      const R* const factors {reinterpret_cast<const R*>(twiddles)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
      vector_begin = width;                                          // its first register's butterflies one at a time: p = 0 multiplies by nothing
      vector_end   = quarter - (quarter % width);                    // whole registers only
      for (std::size_t p {vector_begin}; p < vector_end; p += width) {
        const R* const at        {from + (2 * p)};
        const vec      a         {fft_load(isa, at)};
        const vec      b         {fft_load(isa, at + apart)};
        const vec      c         {fft_load(isa, at + (2 * apart))};
        const vec      d         {fft_load(isa, at + (3 * apart))};
        const vec      a_plus_c  {fft_add(a, c)};
        const vec      a_minus_c {fft_sub(a, c)};
        const vec      b_plus_d  {fft_add(b, d)};
        const vec      i_b_d     {fft_times_i(fft_sub(b, d))};
        const vec      w1        {fft_load(isa, factors + (2 * (p - 1)))};
        const vec      w2        {fft_load(isa, factors + (2 * (run + p - 1)))};
        const vec      w3        {fft_load(isa, factors + (2 * ((2 * run) + p - 1)))};
        fft_store_butterflies(into + (8 * p), fft_add(a_plus_c, b_plus_d),
                              fft_twiddled(fft_real_parts(w1), fft_imaginary_parts(w1), fft_sub(a_minus_c, i_b_d)),
                              fft_twiddled(fft_real_parts(w2), fft_imaginary_parts(w2), fft_sub(a_plus_c, b_plus_d)),
                              fft_twiddled(fft_real_parts(w3), fft_imaginary_parts(w3), fft_add(a_minus_c, i_b_d)));
      }
    }
    for (std::size_t p {0}; p < quarter; p = (p + 1 == vector_begin) ? vector_end : p + 1) { // those the registers did not take
      const bool     twiddle {p != 0};
      const C        w1      {twiddle ? twiddles[p - 1] : C {1}};
      const C        w2      {twiddle ? twiddles[run + p - 1] : C {1}};
      const C        w3      {twiddle ? twiddles[(2 * run) + p - 1] : C {1}};
      const R* const in_p    {from + (2 * stride * p)};
      R* const       to      {into + (2 * stride * 4 * p)};
      std::size_t    q       {0};
      if (stride >= width) {
        const vec w1_re {fft_splat(isa, w1.real())};
        const vec w1_im {fft_splat(isa, w1.imag())};
        const vec w2_re {fft_splat(isa, w2.real())};
        const vec w2_im {fft_splat(isa, w2.imag())};
        const vec w3_re {fft_splat(isa, w3.real())};
        const vec w3_im {fft_splat(isa, w3.imag())};
        for (; q + width <= stride; q += width) {
          const R* const at        {in_p + (2 * q)};
          const vec      a         {fft_load(isa, at)};
          const vec      b         {fft_load(isa, at + apart)};
          const vec      c         {fft_load(isa, at + (2 * apart))};
          const vec      d         {fft_load(isa, at + (3 * apart))};
          const vec      a_plus_c  {fft_add(a, c)};
          const vec      a_minus_c {fft_sub(a, c)};
          const vec      b_plus_d  {fft_add(b, d)};
          const vec      i_b_d     {fft_times_i(fft_sub(b, d))};
          vec            first     {fft_sub(a_minus_c, i_b_d)};
          vec            second    {fft_sub(a_plus_c, b_plus_d)};
          vec            third     {fft_add(a_minus_c, i_b_d)};
          if (twiddle) {
            first  = fft_twiddled(w1_re, w1_im, first);
            second = fft_twiddled(w2_re, w2_im, second);
            third  = fft_twiddled(w3_re, w3_im, third);
          }
          R* const place {to + (2 * q)};
          fft_store(place, fft_add(a_plus_c, b_plus_d));
          fft_store(place + (2 * stride), first);
          fft_store(place + (4 * stride), second);
          fft_store(place + (6 * stride), third);
        }
      }
      for (; q < stride; ++q) { // a stride narrower than a register: one butterfly at a time, the same arithmetic
        const C  a         {in[(stride * p) + q]};
        const C  b         {in[(stride * (p + quarter)) + q]};
        const C  c         {in[(stride * (p + (2 * quarter))) + q]};
        const C  d         {in[(stride * (p + (3 * quarter))) + q]};
        const C  a_plus_c  {a + c};
        const C  a_minus_c {a - c};
        const C  b_plus_d  {b + d};
        const C  b_minus_d {b - d};
        const C  i_b_d     {-b_minus_d.imag(), b_minus_d.real()};
        C* const place     {out + (stride * 4 * p) + q};
        place[0]          = a_plus_c + b_plus_d;
        place[stride]     = twiddle ? fft_fused_product(w1, a_minus_c - i_b_d) : a_minus_c - i_b_d;
        place[2 * stride] = twiddle ? fft_fused_product(w2, a_plus_c - b_plus_d) : a_plus_c - b_plus_d;
        place[3 * stride] = twiddle ? fft_fused_product(w3, a_minus_c + i_b_d) : a_minus_c + i_b_d;
      }
    }
  }

  /*!
   * \brief One Stockham pass of radix 16 of `fft.hpp`'s (`radix_sixteen_pass`), a register of complex numbers of \p Isa
   *        along the stride at a time, which holds whole registers of \p Isa (the first pass is of radix 4, so the stride
   *        is a multiple of 4: AVX2's registers always, AVX-512's of `float` from the second pass of 16): the same sums
   *        in the same order, its factors' products as `fused_product`.
   */
  template <typename Isa,
            typename R>
  NUMERICAL_ALWAYS_INLINE void fft_radix_sixteen_pass(std::size_t                            current,
                                                      std::size_t                            stride,
                                                      const std::complex<R>*                 in,
                                                      std::complex<R>*                       out,
                                                      const std::complex<R>*                 twiddles,
                                                      const std::array<std::complex<R>, 16>& sixteenths) noexcept
  {
    using vec = fft_reg<Isa, R>;
    constexpr Isa         isa   {};
    constexpr std::size_t width {sizeof(vec) / sizeof(std::complex<R>)}; // complex numbers in a register
    const std::size_t     part  {current / 16};
    const std::size_t     run   {part - 1};
    const std::size_t     apart {2 * stride * part};                     // parts between a butterfly's inputs
    const R* const        from  {reinterpret_cast<const R*>(in)};        // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
    R* const              into  {reinterpret_cast<R*>(out)};             // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    for (std::size_t p {0}; p < part; ++p) {
      const R* const in_p {from + (2 * stride * p)};
      R* const       to   {into + (2 * stride * 16 * p)};
      for (std::size_t q {0}; q < stride; q += width) {
        const R* const       at    {in_p + (2 * q)};
        std::array<vec, 16>  inner {};
        NUMERICAL_UNROLL_BLOCK
        for (std::size_t j1 {0}; j1 < 4; ++j1) {
          fft_four_point(fft_load(isa, at + (j1 * apart)), fft_load(isa, at + ((j1 + 4) * apart)), fft_load(isa, at + ((j1 + 8) * apart)),
                         fft_load(isa, at + ((j1 + 12) * apart)), inner[4 * j1], inner[(4 * j1) + 1], inner[(4 * j1) + 2],
                         inner[(4 * j1) + 3]);
        }
        NUMERICAL_UNROLL_BLOCK
        for (std::size_t j1 {1}; j1 < 4; ++j1) {
          NUMERICAL_UNROLL_BLOCK
          for (std::size_t k2 {1}; k2 < 4; ++k2) {
            vec&                   value {inner[(4 * j1) + k2]};
            const std::complex<R>& w     {sixteenths[j1 * k2]};
            value = (j1 * k2) == 4 ? fft_times_minus_i(value) : fft_twiddled(fft_splat(isa, w.real()), fft_splat(isa, w.imag()), value);
          }
        }
        NUMERICAL_UNROLL_BLOCK
        for (std::size_t k2 {0}; k2 < 4; ++k2) {
          std::array<vec, 4>  outputs {};
          fft_four_point(inner[k2], inner[4 + k2], inner[8 + k2], inner[12 + k2], outputs[0], outputs[1], outputs[2],
                         outputs[3]);
          NUMERICAL_UNROLL_BLOCK
          for (std::size_t k1 {0}; k1 < 4; ++k1) {
            const std::size_t k {(4 * k1) + k2};
            vec               y {outputs[k1]};
            if (p != 0 && k != 0) {
              const std::complex<R>& w {twiddles[((k - 1) * run) + p - 1]};
              y = fft_twiddled(fft_splat(isa, w.real()), fft_splat(isa, w.imag()), y);
            }
            fft_store(to + (2 * q) + (2 * stride * k), y);
          }
        }
      }
    }
  }

  //! \brief Output \p k of butterfly \p p of a pass, \p value, stored at \p place: times `w^{kp}` (its factors \p twiddles,
  //!        a run of \p run each power, as \ref avx2_twiddled) unless \p p is 0.
  template <typename Isa,
            typename R>
  NUMERICAL_ALWAYS_INLINE void fft_store_twiddled(R*                     place,
                                                  const fft_reg<Isa, R>& value,
                                                  const std::complex<R>* twiddles,
                                                  std::size_t            run,
                                                  std::size_t            p,
                                                  std::size_t            k) noexcept
  {
    constexpr Isa isa {};
    if (p != 0) {
      const std::complex<R> w {twiddles[((k - 1) * run) + p - 1]};
      fft_store(place, fft_twiddled(fft_splat(isa, w.real()), fft_splat(isa, w.imag()), value));
      return;
    }
    fft_store(place, value);
  }

  /*!
   * \brief One Stockham pass of radix 3 of `fft.hpp`'s (`radix_three_pass_body`), fused, a register of complex numbers
   *        along the stride at a time where it holds one, else one at a time: the same operations in the same order.
   *        \p half_root_three is √3/2.
   */
  template <typename Isa,
            typename R>
  NUMERICAL_ALWAYS_INLINE void fft_radix_three_pass(std::size_t            current,
                                                    std::size_t            stride,
                                                    const std::complex<R>* in,
                                                    std::complex<R>*       out,
                                                    const std::complex<R>* twiddles,
                                                    R                      half_root_three) noexcept
  {
    using C   = std::complex<R>;
    using vec = fft_reg<Isa, R>;
    constexpr Isa         isa   {};
    constexpr std::size_t width {sizeof(vec) / sizeof(std::complex<R>)};
    const std::size_t     part  {current / 3};
    const std::size_t     run   {part - 1};
    const std::size_t     apart {stride * part};
    const vec             half  {fft_splat(isa, R(0.5))};
    const vec             root  {fft_splat(isa, half_root_three)};
    for (std::size_t p {0}; p < part; ++p) {
      const C* const in_p {in + (stride * p)};
      C* const       to   {out + (stride * 3 * p)};
      std::size_t    q    {0};
      if (stride >= width) {
        const C   w1    {p == 0 ? C {1} : twiddles[p - 1]};
        const C   w2    {p == 0 ? C {1} : twiddles[run + p - 1]};
        const vec w1_re {fft_splat(isa, w1.real())};
        const vec w1_im {fft_splat(isa, w1.imag())};
        const vec w2_re {fft_splat(isa, w2.real())};
        const vec w2_im {fft_splat(isa, w2.imag())};
        for (; q + width <= stride; q += width) {
          const R* const at {reinterpret_cast<const R*>(in_p + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
          const vec      x0 {fft_load(isa, at)};
          const vec      x1 {fft_load(isa, at + (2 * apart))};
          const vec      x2 {fft_load(isa, at + (4 * apart))};
          const vec      t  {fft_add(x1, x2)};
          const vec      v  {fft_sub(x0, fft_mul(half, t))};
          const vec      iu {fft_times_i(fft_mul(root, fft_sub(x1, x2)))};
          vec            y1 {fft_sub(v, iu)};
          vec            y2 {fft_add(v, iu)};
          if (p != 0) {
            y1 = fft_twiddled(w1_re, w1_im, y1);
            y2 = fft_twiddled(w2_re, w2_im, y2);
          }
          R* const place {reinterpret_cast<R*>(to + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
          fft_store(place, fft_add(x0, t));
          fft_store(place + (2 * stride), y1);
          fft_store(place + (4 * stride), y2);
        }
      }
      if constexpr (!Isa::whole_strides) {
        for (; q < stride; ++q) { // a stride narrower than a register
          const C x0 {in_p[q]};
          const C x1 {in_p[q + apart]};
          const C x2 {in_p[q + (2 * apart)]};
          const C t  {x1 + x2};
          const C v  {x0 - C {R(0.5) * t.real(), R(0.5) * t.imag()}};
          const C d  {x1 - x2};
          const C u  {half_root_three * d.real(), half_root_three * d.imag()};
          const C iu {-u.imag(), u.real()};
          to[q]              = x0 + t;
          to[q + stride]     = p == 0 ? v - iu : fft_fused_product(twiddles[p - 1], v - iu);
          to[q + 2 * stride] = p == 0 ? v + iu : fft_fused_product(twiddles[run + p - 1], v + iu);
        }
      }
    }
  }

  /*!
   * \brief One Stockham pass of radix 5 of `fft.hpp`'s (`radix_five_pass_body`), fused, as \ref fft_radix_three_pass:
   *        \p c1, \p c2 the cosines and \p s1, \p s2 the sines of 2π/5 and 4π/5.
   */
  template <typename Isa,
            typename R>
  NUMERICAL_ALWAYS_INLINE void fft_radix_five_pass(std::size_t            current,
                                                   std::size_t            stride,
                                                   const std::complex<R>* in,
                                                   std::complex<R>*       out,
                                                   const std::complex<R>* twiddles,
                                                   R                      c1,
                                                   R                      c2,
                                                   R                      s1,
                                                   R                      s2) noexcept
  {
    using C   = std::complex<R>;
    using vec = fft_reg<Isa, R>;
    constexpr Isa         isa   {};
    constexpr std::size_t width {sizeof(vec) / sizeof(std::complex<R>)};
    const std::size_t     part  {current / 5};
    const std::size_t     run   {part - 1};
    const std::size_t     apart {stride * part};
    const vec             vc1   {fft_splat(isa, c1)};
    const vec             vc2   {fft_splat(isa, c2)};
    const vec             vs1   {fft_splat(isa, s1)};
    const vec             vs2   {fft_splat(isa, s2)};
    const vec             vms1  {fft_splat(isa, -s1)};
    for (std::size_t p {0}; p < part; ++p) {
      const C* const in_p {in + (stride * p)};
      C* const       to   {out + (stride * 5 * p)};
      std::size_t    q    {0};
      if (stride >= width) {
        for (; q + width <= stride; q += width) {
          const R* const at    {reinterpret_cast<const R*>(in_p + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
          const vec      x0    {fft_load(isa, at)};
          const vec      x1    {fft_load(isa, at + (2 * apart))};
          const vec      x2    {fft_load(isa, at + (4 * apart))};
          const vec      x3    {fft_load(isa, at + (6 * apart))};
          const vec      x4    {fft_load(isa, at + (8 * apart))};
          const vec      t1    {fft_add(x1, x4)};
          const vec      t2    {fft_add(x2, x3)};
          const vec      t3    {fft_sub(x1, x4)};
          const vec      t4    {fft_sub(x2, x3)};
          const vec      a1    {fft_fused(vc2, t2, fft_fused(vc1, t1, x0))};
          const vec      a2    {fft_fused(vc1, t2, fft_fused(vc2, t1, x0))};
          const vec      ib1   {fft_times_i(fft_fused(vs2, t4, fft_mul(vs1, t3)))};
          const vec      ib2   {fft_times_i(fft_fused(vms1, t4, fft_mul(vs2, t3)))};
          R* const       place {reinterpret_cast<R*>(to + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
          fft_store(place, fft_add(fft_add(x0, t1), t2));
          fft_store_twiddled<Isa>(place + (2 * stride), fft_sub(a1, ib1), twiddles, run, p, 1);
          fft_store_twiddled<Isa>(place + (4 * stride), fft_sub(a2, ib2), twiddles, run, p, 2);
          fft_store_twiddled<Isa>(place + (6 * stride), fft_add(a2, ib2), twiddles, run, p, 3);
          fft_store_twiddled<Isa>(place + (8 * stride), fft_add(a1, ib1), twiddles, run, p, 4);
        }
      }
      if constexpr (!Isa::whole_strides) {
        for (; q < stride; ++q) { // a stride narrower than a register
          const C                x0  {in_p[q]};
          const C                x1  {in_p[q + apart]};
          const C                x2  {in_p[q + (2 * apart)]};
          const C                x3  {in_p[q + (3 * apart)]};
          const C                x4  {in_p[q + (4 * apart)]};
          const C                t1  {x1 + x4};
          const C                t2  {x2 + x3};
          const C                t3  {x1 - x4};
          const C                t4  {x2 - x3};
          const C                a1  {fft_scaled_add(fft_scaled_add(x0, c1, t1), c2, t2)};
          const C                a2  {fft_scaled_add(fft_scaled_add(x0, c2, t1), c1, t2)};
          const C                b1  {fft_scaled_add(C {s1 * t3.real(), s1 * t3.imag()}, s2, t4)};
          const C                b2  {fft_scaled_add(C {s2 * t3.real(), s2 * t3.imag()}, -s1, t4)};
          const C                ib1 {-b1.imag(), b1.real()};
          const C                ib2 {-b2.imag(), b2.real()};
          const std::array<C, 4> y   {a1 - ib1, a2 - ib2, a2 + ib2, a1 + ib1};
          to[q] = (x0 + t1) + t2;
          for (std::size_t j {1}; j < 5; ++j) {
            to[q + (j * stride)] = p == 0 ? y[j - 1] : fft_fused_product(twiddles[((j - 1) * run) + p - 1], y[j - 1]);
          }
        }
      }
    }
  }

  /*!
   * \brief One Stockham pass of radix 8 of `fft.hpp`'s (`radix_eight_pass_body`), fused, a register of complex numbers
   *        along the stride at a time (the stride a power of two of at least 4: powers of two run first), the same
   *        operations in the same order. \p half_root_two is √½.
   */
  template <typename Isa,
            typename R>
  NUMERICAL_ALWAYS_INLINE void fft_radix_eight_pass(std::size_t            current,
                                                    std::size_t            stride,
                                                    const std::complex<R>* in,
                                                    std::complex<R>*       out,
                                                    const std::complex<R>* twiddles,
                                                    R                      half_root_two) noexcept
  {
    using vec = fft_reg<Isa, R>;
    constexpr Isa         isa   {};
    constexpr std::size_t width {sizeof(vec) / sizeof(std::complex<R>)};
    const std::size_t     part  {current / 8};
    const std::size_t     run   {part - 1};
    const std::size_t     apart {2 * stride * part};              // parts between a butterfly's inputs
    const vec             root  {fft_splat(isa, half_root_two)};
    const R* const        from  {reinterpret_cast<const R*>(in)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
    R* const              into  {reinterpret_cast<R*>(out)};      // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    for (std::size_t p {0}; p < part; ++p) {
      const R* const in_p {from + (2 * stride * p)};
      R* const       to   {into + (2 * stride * 8 * p)};
      for (std::size_t q {0}; q + width <= stride; q += width) {
        const R* const at {in_p + (2 * q)};
        const vec      x0 {fft_load(isa, at)};
        const vec      x1 {fft_load(isa, at + apart)};
        const vec      x2 {fft_load(isa, at + (2 * apart))};
        const vec      x3 {fft_load(isa, at + (3 * apart))};
        const vec      x4 {fft_load(isa, at + (4 * apart))};
        const vec      x5 {fft_load(isa, at + (5 * apart))};
        const vec      x6 {fft_load(isa, at + (6 * apart))};
        const vec      x7 {fft_load(isa, at + (7 * apart))};
        const vec      b1 {fft_sub(x1, x5)};
        const vec      b3 {fft_sub(x3, x7)};
        const vec      s1 {fft_add(b1, fft_exchanged(b1))}; // re + im in both lanes
        const vec      d1 {fft_sub(b1, fft_exchanged(b1))}; // re − im, im − re
        const vec      s3 {fft_add(b3, fft_exchanged(b3))};
        const vec      d3 {fft_sub(fft_exchanged(b3), b3)}; // im − re, re − im
        const vec      w1 {fft_mul(root, fft_parts_of(s1, d1))};
        const vec      w3 {fft_conjugated(fft_mul(root, fft_parts_of(d3, s3)))};
        vec            e0 {};
        vec            e1 {};
        vec            e2 {};
        vec            e3 {};
        vec            o0 {};
        vec            o1 {};
        vec            o2 {};
        vec            o3 {};
        fft_four_point(fft_add(x0, x4), fft_add(x1, x5), fft_add(x2, x6), fft_add(x3, x7), e0, e1, e2, e3);
        fft_four_point(fft_sub(x0, x4), w1, fft_times_minus_i(fft_sub(x2, x6)), w3, o0, o1, o2, o3);
        R* const place {to + (2 * q)};
        fft_store(place, e0);
        fft_store_twiddled<Isa>(place + (2 * stride), o0, twiddles, run, p, 1);
        fft_store_twiddled<Isa>(place + (4 * stride), e1, twiddles, run, p, 2);
        fft_store_twiddled<Isa>(place + (6 * stride), o1, twiddles, run, p, 3);
        fft_store_twiddled<Isa>(place + (8 * stride), e2, twiddles, run, p, 4);
        fft_store_twiddled<Isa>(place + (10 * stride), o2, twiddles, run, p, 5);
        fft_store_twiddled<Isa>(place + (12 * stride), e3, twiddles, run, p, 6);
        fft_store_twiddled<Isa>(place + (14 * stride), o3, twiddles, run, p, 7);
      }
    }
  }

#endif

#if defined(NUMERICAL_X86_COMPILED)
  //! \brief \ref fft_radix_four_pass of AVX2.
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_radix_four_pass(std::size_t            current,
                                                                std::size_t            stride,
                                                                const std::complex<R>* in,
                                                                std::complex<R>*       out,
                                                                const std::complex<R>* twiddles) noexcept
  {
    fft_radix_four_pass<fft_avx2>(current, stride, in, out, twiddles);
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref fft_radix_four_pass of AVX-512.
  template <typename R>
  NUMERICAL_TARGET("avx512f,avx2,fma") inline void avx512_radix_four_pass(std::size_t            current,
                                                                          std::size_t            stride,
                                                                          const std::complex<R>* in,
                                                                          std::complex<R>*       out,
                                                                          const std::complex<R>* twiddles) noexcept
  {
    fft_radix_four_pass<fft_avx512>(current, stride, in, out, twiddles);
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref fft_radix_sixteen_pass of AVX2.
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_radix_sixteen_pass(std::size_t                            current,
                                                                   std::size_t                            stride,
                                                                   const std::complex<R>*                 in,
                                                                   std::complex<R>*                       out,
                                                                   const std::complex<R>*                 twiddles,
                                                                   const std::array<std::complex<R>, 16>& sixteenths) noexcept
  {
    fft_radix_sixteen_pass<fft_avx2>(current, stride, in, out, twiddles, sixteenths);
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref fft_radix_sixteen_pass of AVX-512 where the stride holds its registers, else of AVX2 (the same bits).
  template <typename R>
  NUMERICAL_TARGET("avx512f,avx2,fma") inline void avx512_radix_sixteen_pass(std::size_t                            current,
                                                                             std::size_t                            stride,
                                                                             const std::complex<R>*                 in,
                                                                             std::complex<R>*                       out,
                                                                             const std::complex<R>*                 twiddles,
                                                                             const std::array<std::complex<R>, 16>& sixteenths) noexcept
  {
    if (stride % (sizeof(fft_reg<fft_avx512, R>) / sizeof(std::complex<R>)) == 0) {
      fft_radix_sixteen_pass<fft_avx512>(current, stride, in, out, twiddles, sixteenths);
    }
    else {
      fft_radix_sixteen_pass<fft_avx2>(current, stride, in, out, twiddles, sixteenths);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref fft_radix_three_pass of AVX-512, its stride a multiple of a register's complex numbers.
  template <typename R>
  NUMERICAL_TARGET("avx512f,avx2,fma") inline void avx512_radix_three_pass(std::size_t            current,
                                                                           std::size_t            stride,
                                                                           const std::complex<R>* in,
                                                                           std::complex<R>      * out,
                                                                           const std::complex<R>* twiddles,
                                                                           R                      half_root_three) noexcept
  {
    fft_radix_three_pass<fft_avx512>(current, stride, in, out, twiddles, half_root_three);
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref fft_radix_five_pass of AVX-512, its stride a multiple of a register's complex numbers.
  template <typename R>
  NUMERICAL_TARGET("avx512f,avx2,fma") inline void avx512_radix_five_pass(std::size_t            current,
                                                                          std::size_t            stride,
                                                                          const std::complex<R>* in,
                                                                          std::complex<R>      * out,
                                                                          const std::complex<R>* twiddles,
                                                                          R                      c1,
                                                                          R                      c2,
                                                                          R                      s1,
                                                                          R                      s2) noexcept
  {
    fft_radix_five_pass<fft_avx512>(current, stride, in, out, twiddles, c1, c2, s1, s2);
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref fft_radix_eight_pass of AVX-512, its stride a multiple of a register's complex numbers.
  template <typename R>
  NUMERICAL_TARGET("avx512f,avx2,fma") inline void avx512_radix_eight_pass(std::size_t            current,
                                                                           std::size_t            stride,
                                                                           const std::complex<R>* in,
                                                                           std::complex<R>      * out,
                                                                           const std::complex<R>* twiddles,
                                                                           R                      half_root_two) noexcept
  {
    fft_radix_eight_pass<fft_avx512>(current, stride, in, out, twiddles, half_root_two);
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

#endif

#if defined(NUMERICAL_NEON_COMPILED)

  //! \brief \ref fft_radix_four_pass of NEON.
  template <typename R>
  void neon_radix_four_pass(std::size_t            current,
                            std::size_t            stride,
                            const std::complex<R>* in,
                            std::complex<R>*       out,
                            const std::complex<R>* twiddles) noexcept
  {
    fft_radix_four_pass<fft_neon>(current, stride, in, out, twiddles);
  }

  //! \brief \ref fft_radix_sixteen_pass of NEON (its registers hold one or two complex numbers: every stride's).
  template <typename R>
  void neon_radix_sixteen_pass(std::size_t                            current,
                               std::size_t                            stride,
                               const std::complex<R>*                 in,
                               std::complex<R>*                       out,
                               const std::complex<R>*                 twiddles,
                               const std::array<std::complex<R>, 16>& sixteenths) noexcept
  {
    fft_radix_sixteen_pass<fft_neon>(current, stride, in, out, twiddles, sixteenths);
  }

  //! \brief \ref fft_radix_three_pass of NEON.
  template <typename R>
  void neon_radix_three_pass(std::size_t            current,
                             std::size_t            stride,
                             const std::complex<R>* in,
                             std::complex<R>      * out,
                             const std::complex<R>* twiddles,
                             R                      half_root_three) noexcept
  {
    fft_radix_three_pass<fft_neon>(current, stride, in, out, twiddles, half_root_three);
  }

  //! \brief \ref fft_radix_five_pass of NEON.
  template <typename R>
  void neon_radix_five_pass(std::size_t            current,
                            std::size_t            stride,
                            const std::complex<R>* in,
                            std::complex<R>      * out,
                            const std::complex<R>* twiddles,
                            R                      c1,
                            R                      c2,
                            R                      s1,
                            R                      s2) noexcept
  {
    fft_radix_five_pass<fft_neon>(current, stride, in, out, twiddles, c1, c2, s1, s2);
  }

  //! \brief \ref fft_radix_eight_pass of NEON.
  template <typename R>
  void neon_radix_eight_pass(std::size_t            current,
                             std::size_t            stride,
                             const std::complex<R>* in,
                             std::complex<R>      * out,
                             const std::complex<R>* twiddles,
                             R                      half_root_two) noexcept
  {
    fft_radix_eight_pass<fft_neon>(current, stride, in, out, twiddles, half_root_two);
  }

#endif
} // namespace numerical::detail

#endif // NUMERICAL_FFT_KERNELS_HPP
