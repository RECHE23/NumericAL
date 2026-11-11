/*!
 * \file elementary.hpp
 * \brief `exp` and `expm1` of `double` by a table of 2^(j/128) and a polynomial, branch-free so that a loop of them
 *        runs in vectors: within 0.506 ulp of the exact value for a normal result of `exp`, 0.503 ulp for `expm1`; the
 *        same bits from a vector lane as from a call.
 */
#ifndef NUMERICAL_ELEMENTARY_HPP
#define NUMERICAL_ELEMENTARY_HPP

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "simd.hpp"

#if defined(__aarch64__) && defined(__GNUC__)
#  define NUMERICAL_EXP_NEON // table_exp of two lanes in NEON (tested with #ifdef only)
#  include <arm_neon.h>
#endif

namespace numerical::detail {

  //! \brief 128/ln 2 (`tools/exp_table.py` derives each constant here).
  inline constexpr double exp_inv_ln2_128 {0x1.71547652b82fep+7};

  //! \brief ln 2/128 to 32 significant bits, toward zero: its product by an integer of 18 bits or fewer is exact, and
  //!        so is that product's difference from an argument it reduces.
  inline constexpr double exp_ln2_128_high {0x1.62e42fee00000p-8};

  //! \brief What \ref exp_ln2_128_high leaves out of ln 2/128, to the nearest `double`.
  inline constexpr double exp_ln2_128_low {0x1.a39ef35793c76p-40};

  //! \brief ln 2^-1022 toward zero: below it, `exp` is subnormal, where \ref table_exp's two scalings would round twice.
  inline constexpr double exp_normal_bound {-0x1.6232bdd7abcd2p+9};

  //! \brief 2^(j/128) for j = 0…127, at 2j the `double` nearest it and at 2j + 1 the `double` nearest what that one
  //!        leaves out. Not `inline`: a copy in each translation unit, whose loads Clang's vectoriser bounds, where it
  //!        kept a loop of \ref table_exp scalar over an inline variable's ("cannot identify array bounds").
  constexpr std::array<double, 256> exp_table {
    0x1.0000000000000p+0, 0x0.0p+0,
    0x1.0163da9fb3335p+0, 0x1.b61299ab8cdb7p-54,
    0x1.02c9a3e778061p+0, -0x1.19083535b085dp-56,
    0x1.04315e86e7f85p+0, -0x1.0a31c1977c96ep-54,
    0x1.059b0d3158574p+0, 0x1.d73e2a475b465p-55,
    0x1.0706b29ddf6dep+0, -0x1.c91dfe2b13c27p-55,
    0x1.0874518759bc8p+0, 0x1.186be4bb284ffp-57,
    0x1.09e3ecac6f383p+0, 0x1.1487818316136p-54,
    0x1.0b5586cf9890fp+0, 0x1.8a62e4adc610bp-54,
    0x1.0cc922b7247f7p+0, 0x1.01edc16e24f71p-54,
    0x1.0e3ec32d3d1a2p+0, 0x1.03a1727c57b53p-59,
    0x1.0fb66affed31bp+0, -0x1.b9bedc44ebd7bp-57,
    0x1.11301d0125b51p+0, -0x1.6c51039449b3ap-54,
    0x1.12abdc06c31ccp+0, -0x1.1b514b36ca5c7p-58,
    0x1.1429aaea92de0p+0, -0x1.32fbf9af1369ep-54,
    0x1.15a98c8a58e51p+0, 0x1.2406ab9eeab0ap-55,
    0x1.172b83c7d517bp+0, -0x1.19041b9d78a76p-55,
    0x1.18af9388c8deap+0, -0x1.11023d1970f6cp-54,
    0x1.1a35beb6fcb75p+0, 0x1.e5b4c7b4968e4p-55,
    0x1.1bbe084045cd4p+0, -0x1.95386352ef607p-54,
    0x1.1d4873168b9aap+0, 0x1.e016e00a2643cp-54,
    0x1.1ed5022fcd91dp+0, -0x1.1df98027bb78cp-54,
    0x1.2063b88628cd6p+0, 0x1.dc775814a8495p-55,
    0x1.21f49917ddc96p+0, 0x1.2a97e9494a5eep-55,
    0x1.2387a6e756238p+0, 0x1.9b07eb6c70573p-54,
    0x1.251ce4fb2a63fp+0, 0x1.ac155bef4f4a4p-55,
    0x1.26b4565e27cddp+0, 0x1.2bd339940e9d9p-55,
    0x1.284dfe1f56381p+0, -0x1.a4c3a8c3f0d7ep-54,
    0x1.29e9df51fdee1p+0, 0x1.612e8afad1255p-55,
    0x1.2b87fd0dad990p+0, -0x1.10adcd6381aa4p-59,
    0x1.2d285a6e4030bp+0, 0x1.0024754db41d5p-54,
    0x1.2ecafa93e2f56p+0, 0x1.1ca0f45d52383p-56,
    0x1.306fe0a31b715p+0, 0x1.6f46ad23182e4p-55,
    0x1.32170fc4cd831p+0, 0x1.a9ce78e18047cp-55,
    0x1.33c08b26416ffp+0, 0x1.32721843659a6p-54,
    0x1.356c55f929ff1p+0, -0x1.b5cee5c4e4628p-55,
    0x1.371a7373aa9cbp+0, -0x1.63aeabf42eae2p-54,
    0x1.38cae6d05d866p+0, -0x1.e958d3c9904bdp-54,
    0x1.3a7db34e59ff7p+0, -0x1.5e436d661f5e3p-56,
    0x1.3c32dc313a8e5p+0, -0x1.efff8375d29c3p-54,
    0x1.3dea64c123422p+0, 0x1.ada0911f09ebcp-55,
    0x1.3fa4504ac801cp+0, -0x1.7d023f956f9f3p-54,
    0x1.4160a21f72e2ap+0, -0x1.ef3691c309278p-58,
    0x1.431f5d950a897p+0, -0x1.1c7dde35f7999p-55,
    0x1.44e086061892dp+0, 0x1.89b7a04ef80d0p-59,
    0x1.46a41ed1d0057p+0, 0x1.c944bd1648a76p-54,
    0x1.486a2b5c13cd0p+0, 0x1.3c1a3b69062f0p-56,
    0x1.4a32af0d7d3dep+0, 0x1.9cb62f3d1be56p-54,
    0x1.4bfdad5362a27p+0, 0x1.d4397afec42e2p-56,
    0x1.4dcb299fddd0dp+0, 0x1.8ecdbbc6a7833p-54,
    0x1.4f9b2769d2ca7p+0, -0x1.4b309d25957e3p-54,
    0x1.516daa2cf6642p+0, -0x1.f768569bd93efp-55,
    0x1.5342b569d4f82p+0, -0x1.07abe1db13cadp-55,
    0x1.551a4ca5d920fp+0, -0x1.d689cefede59bp-55,
    0x1.56f4736b527dap+0, 0x1.9bb2c011d93adp-54,
    0x1.58d12d497c7fdp+0, 0x1.295e15b9a1de8p-55,
    0x1.5ab07dd485429p+0, 0x1.6324c054647adp-54,
    0x1.5c9268a5946b7p+0, 0x1.c4b1b816986a2p-60,
    0x1.5e76f15ad2148p+0, 0x1.ba6f93080e65ep-54,
    0x1.605e1b976dc09p+0, -0x1.3e2429b56de47p-54,
    0x1.6247eb03a5585p+0, -0x1.383c17e40b497p-54,
    0x1.6434634ccc320p+0, -0x1.c483c759d8933p-55,
    0x1.6623882552225p+0, -0x1.bb60987591c34p-54,
    0x1.68155d44ca973p+0, 0x1.038ae44f73e65p-57,
    0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54,
    0x1.6c012750bdabfp+0, -0x1.2895667ff0b0dp-56,
    0x1.6dfb23c651a2fp+0, -0x1.bbe3a683c88abp-57,
    0x1.6ff7df9519484p+0, -0x1.83c0f25860ef6p-55,
    0x1.71f75e8ec5f74p+0, -0x1.16e4786887a99p-55,
    0x1.73f9a48a58174p+0, -0x1.0a8d96c65d53cp-54,
    0x1.75feb564267c9p+0, -0x1.0245957316dd3p-54,
    0x1.780694fde5d3fp+0, 0x1.866b80a02162dp-54,
    0x1.7a11473eb0187p+0, -0x1.41577ee04992fp-55,
    0x1.7c1ed0130c132p+0, 0x1.f124cd1164dd6p-54,
    0x1.7e2f336cf4e62p+0, 0x1.05d02ba15797ep-56,
    0x1.80427543e1a12p+0, -0x1.27c86626d972bp-54,
    0x1.82589994cce13p+0, -0x1.d4c1dd41532d8p-54,
    0x1.8471a4623c7adp+0, -0x1.8d684a341cdfbp-55,
    0x1.868d99b4492edp+0, -0x1.fc6f89bd4f6bap-54,
    0x1.88ac7d98a6699p+0, 0x1.994c2f37cb53ap-54,
    0x1.8ace5422aa0dbp+0, 0x1.6e9f156864b27p-54,
    0x1.8cf3216b5448cp+0, -0x1.0d55e32e9e3aap-56,
    0x1.8f1ae99157736p+0, 0x1.5cc13a2e3976cp-55,
    0x1.9145b0b91ffc6p+0, -0x1.dd6792e582524p-54,
    0x1.93737b0cdc5e5p+0, -0x1.75fc781b57ebcp-57,
    0x1.95a44cbc8520fp+0, -0x1.64b7c96a5f039p-56,
    0x1.97d829fde4e50p+0, -0x1.d185b7c1b85d1p-54,
    0x1.9a0f170ca07bap+0, -0x1.173bd91cee632p-54,
    0x1.9c49182a3f090p+0, 0x1.c7c46b071f2bep-56,
    0x1.9e86319e32323p+0, 0x1.824ca78e64c6ep-56,
    0x1.a0c667b5de565p+0, -0x1.359495d1cd533p-54,
    0x1.a309bec4a2d33p+0, 0x1.6305c7ddc36abp-54,
    0x1.a5503b23e255dp+0, -0x1.d2f6edb8d41e1p-54,
    0x1.a799e1330b358p+0, 0x1.bcb7ecac563c7p-54,
    0x1.a9e6b5579fdbfp+0, 0x1.0fac90ef7fd31p-54,
    0x1.ac36bbfd3f37ap+0, -0x1.f9234cae76cd0p-55,
    0x1.ae89f995ad3adp+0, 0x1.7a1cd345dcc81p-54,
    0x1.b0e07298db666p+0, -0x1.bdef54c80e425p-54,
    0x1.b33a2b84f15fbp+0, -0x1.2805e3084d708p-57,
    0x1.b59728de5593ap+0, -0x1.c71dfbbba6de3p-54,
    0x1.b7f76f2fb5e47p+0, -0x1.5584f7e54ac3bp-56,
    0x1.ba5b030a1064ap+0, -0x1.efcd30e54292ep-54,
    0x1.bcc1e904bc1d2p+0, 0x1.23dd07a2d9e84p-55,
    0x1.bf2c25bd71e09p+0, -0x1.efdca3f6b9c73p-54,
    0x1.c199bdd85529cp+0, 0x1.11065895048ddp-55,
    0x1.c40ab5fffd07ap+0, 0x1.b4537e083c60ap-54,
    0x1.c67f12e57d14bp+0, 0x1.2884dff483cadp-54,
    0x1.c8f6d9406e7b5p+0, 0x1.1acbc48805c44p-56,
    0x1.cb720dcef9069p+0, 0x1.503cbd1e949dbp-56,
    0x1.cdf0b555dc3fap+0, -0x1.dd83b53829d72p-55,
    0x1.d072d4a07897cp+0, -0x1.cbc3743797a9cp-54,
    0x1.d2f87080d89f2p+0, -0x1.d487b719d8578p-54,
    0x1.d5818dcfba487p+0, 0x1.2ed02d75b3707p-55,
    0x1.d80e316c98398p+0, -0x1.11ec18beddfe8p-54,
    0x1.da9e603db3285p+0, 0x1.c2300696db532p-54,
    0x1.dd321f301b460p+0, 0x1.2da5778f018c3p-54,
    0x1.dfc97337b9b5fp+0, -0x1.1a5cd4f184b5cp-54,
    0x1.e264614f5a129p+0, -0x1.7b627817a1496p-54,
    0x1.e502ee78b3ff6p+0, 0x1.39e8980a9cc8fp-55,
    0x1.e7a51fbc74c83p+0, 0x1.2d522ca0c8de2p-54,
    0x1.ea4afa2a490dap+0, -0x1.e9c23179c2893p-54,
    0x1.ecf482d8e67f1p+0, -0x1.c93f3b411ad8cp-54,
    0x1.efa1bee615a27p+0, 0x1.dc7f486a4b6b0p-54,
    0x1.f252b376bba97p+0, 0x1.3a1a5bf0d8e43p-54,
    0x1.f50765b6e4540p+0, 0x1.9d3e12dd8a18bp-54,
    0x1.f7bfdad9cbe14p+0, -0x1.dbb12d006350ap-54,
    0x1.fa7c1819e90d8p+0, 0x1.74853f3a5931ep-55,
    0x1.fd3c22b8f71f1p+0, 0x1.2eb74966579e7p-57,
  };

  /*!
   * \brief exp(\p x) for an \p x whose result is normal or infinite (\ref exp_normal_bound or above): x = (128 e + j)
   *        ln 2/128 + r, |r| ≤ ln 2/256, and exp(x) = 2^e · 2^(j/128) · exp(r), exp(r) − 1 by its Taylor polynomial of
   *        degree 6. Every multiply-add `std::fma`, written out, so that no compiler fuses otherwise; inlined into the
   *        loop that calls it (\ref exp_into). Below the bound, its answer is not exp's; for NaN, NaN.
   */
  NUMERICAL_ALWAYS_INLINE double table_exp(double x) noexcept
  {
    constexpr double shifter {0x1.8p52};              // n in the low bits of x·128/ln 2 + shifter, rounded to an integer
    const double     clamped {x > 710.0 ? 710.0 : x}; // past it, the result overflows all the same
    const double     z       {std::fma(clamped, exp_inv_ln2_128, shifter)};
    const auto       n       {std::bit_cast<std::int64_t>(z) - std::bit_cast<std::int64_t>(shifter)};
    const double     k       {z - shifter};
    const double     r       {std::fma(-k, exp_ln2_128_low, std::fma(-k, exp_ln2_128_high, clamped))};
    const double     r2      {r * r};
    double           p       {std::fma(r, 1.0 / 720.0, 1.0 / 120.0)};
    p = std::fma(r, p, 1.0 / 24.0);
    p = std::fma(r, p, 1.0 / 6.0);
    p = std::fma(r, p, 0.5);
    p = std::fma(r2, p, r);                                                                     // exp(r) − 1
    const auto         j      {static_cast<std::size_t>(static_cast<std::uint64_t>(n) & 127U)}; // n mod 128
    const double       head   {exp_table[2 * j]};
    const double       tail   {exp_table[(2 * j) + 1]};
    const double       scaled {head + std::fma(head + tail, p, tail)};
    // 2^e in two factors, each a normal double for e from −1023 to 1024 (the bound and the clamp): 2^1024 overflows
    const std::int64_t e      {(n - static_cast<std::int64_t>(j)) / 128}; // exact: n − j is a multiple of 128
    const std::int64_t half   {e / 2};
    const double       first  {std::bit_cast<double>(static_cast<std::uint64_t>(half + 1023) << 52U)};
    const double       second {std::bit_cast<double>(static_cast<std::uint64_t>(e - half + 1023) << 52U)};
    const double       result {(scaled * first) * second};
    return std::isnan(x) ? x : result;
  }

  //! \brief The sum of \p a and \p b as `sum` rounded and `error` exact (Knuth's two-sum: no order of magnitudes needed).
  NUMERICAL_ALWAYS_INLINE void two_sum(double  a,
                                       double  b,
                                       double& sum,
                                       double& error) noexcept
  {
    sum = a + b;
    const double b_part {sum - a};
    error = (a - (sum - b_part)) + (b - b_part);
  }

  /*!
   * \brief expm1(\p x) by \ref table_exp's reduction, x = (128 e + j) ln 2/128 + r: in units of 2^e, 2^(j/128) − 2^−e
   *        + 2^(j/128) r_head, each sum and product of them exact (two-sums and a fused product's error), plus the
   *        small rest, 2^(j/128) (r − r_head + exp(r) − 1 − r) and the table's tail times exp(r), then scaled by 2^e: the
   *        cancellation near 0 and ±ln 2 leaves no rounding but the last. Within 0.503 ulp of the exact value; below
   *        −40, −1 (exp(x) under half an ulp of it); x itself where |x| < 2^−54 (±0 kept) and for NaN.
   */
  NUMERICAL_ALWAYS_INLINE double table_expm1(double x) noexcept
  {
    constexpr double shifter {0x1.8p52};
    const double     clamped {x > 710.0 ? 710.0 : (x < -40.0 ? -40.0 : x)};
    const double     z       {std::fma(clamped, exp_inv_ln2_128, shifter)};
    const auto       n       {std::bit_cast<std::int64_t>(z) - std::bit_cast<std::int64_t>(shifter)};
    const double     k       {z - shifter};
    const double     r_head  {std::fma(-k, exp_ln2_128_high, clamped)}; // exact, as table_exp's first step
    const double     r       {std::fma(-k, exp_ln2_128_low, r_head)};
    double           q       {std::fma(r, 1.0 / 5040.0, 1.0 / 720.0)};
    q = std::fma(r, q, 1.0 / 120.0);
    q = std::fma(r, q, 1.0 / 24.0);
    q = std::fma(r, q, 1.0 / 6.0);
    q = std::fma(r, q, 0.5);
    // every product beside a sum fused explicitly, or used by no sum, so that no compiler's contraction rounds the
    // scalar call and the vector loop two ways
    const double       square    {r * r};
    const double       rest      {square * q};                          // exp(r) − 1 − r
    const double       p         {std::fma(square, q, r)};              // exp(r) − 1
    const double       r_rest    {std::fma(-k, exp_ln2_128_low, rest)}; // r − r_head + exp(r) − 1 − r
    const auto         j         {static_cast<std::size_t>(static_cast<std::uint64_t>(n) & 127U)};
    const double       head      {exp_table[2 * j]};
    const double       tail      {exp_table[(2 * j) + 1]};
    const std::int64_t e         {(n - static_cast<std::int64_t>(j)) / 128}; // −58 to 1024
    // 2^−e, a normal double: past 2^−1022, where it is far under an ulp of the head, 2^−1022 in its place
    const double       unit      {std::bit_cast<double>(static_cast<std::uint64_t>(1023 - (e > 1022 ? 1022 : e)) << 52U)};
    double             lead      {};
    double             lead_err  {};
    two_sum(head, -unit, lead, lead_err);
    const double       product   {head * r_head};
    const double       prod_err  {std::fma(head, r_head, -product)};
    double             sum       {};
    double             sum_err   {};
    two_sum(lead, product, sum, sum_err);
    const double       small     {lead_err + sum_err + prod_err + std::fma(head, r_rest, std::fma(tail, p, tail))};
    const std::int64_t half      {e / 2};
    const double       first     {std::bit_cast<double>(static_cast<std::uint64_t>(half + 1023) << 52U)};
    const double       second    {std::bit_cast<double>(static_cast<std::uint64_t>(e - half + 1023) << 52U)};
    const double       result    {((sum + small) * first) * second};
    return std::fabs(x) >= 0x1p-54 ? result : x; // NaN fails the comparison
  }

#ifdef NUMERICAL_EXP_NEON
  /*!
   * \brief \ref table_exp of each lane, in NEON: each operation the scalar one's (`vfmaq_f64` and `vfmsq_f64` its
   *        fused multiply-adds, the table read lane by lane), so each lane the scalar call's bits. Written out, where
   *        Apple's Clang 16 and GCC 14 kept the loop scalar (the table's reads, which NEON gathers lane by lane, priced
   *        above the gain): 10⁶ values on one arm64 core 2.3 → 1.6 ms (Apple's Clang), 2.2 → 1.5 ms (GCC 14).
   */
  NUMERICAL_ALWAYS_INLINE float64x2_t table_exp(float64x2_t x) noexcept
  {
    const float64x2_t shifter {vdupq_n_f64(0x1.8p52)};
    const float64x2_t clamped {vminq_f64(x, vdupq_n_f64(710.0))}; // NaN stays NaN
    const float64x2_t z       {vfmaq_f64(shifter, clamped, vdupq_n_f64(exp_inv_ln2_128))};
    const int64x2_t   n       {vsubq_s64(vreinterpretq_s64_f64(z), vreinterpretq_s64_f64(shifter))};
    const float64x2_t k       {vsubq_f64(z, shifter)};
    const float64x2_t r       {vfmsq_f64(vfmsq_f64(clamped, k, vdupq_n_f64(exp_ln2_128_high)), k, vdupq_n_f64(exp_ln2_128_low))};
    const float64x2_t r2      {vmulq_f64(r, r)};
    float64x2_t       p       {vfmaq_f64(vdupq_n_f64(1.0 / 120.0), r, vdupq_n_f64(1.0 / 720.0))};
    p = vfmaq_f64(vdupq_n_f64(1.0 / 24.0), r, p);
    p = vfmaq_f64(vdupq_n_f64(1.0 / 6.0), r, p);
    p = vfmaq_f64(vdupq_n_f64(0.5), r, p);
    p = vfmaq_f64(r, r2, p);
    const int64x2_t   j      {vandq_s64(n, vdupq_n_s64(127))};
    const float64x2_t low    {vld1q_f64(exp_table.data() + (2 * static_cast<std::size_t>(vgetq_lane_s64(j, 0))))};
    const float64x2_t high   {vld1q_f64(exp_table.data() + (2 * static_cast<std::size_t>(vgetq_lane_s64(j, 1))))};
    const float64x2_t head   {vzip1q_f64(low, high)};
    const float64x2_t tail   {vzip2q_f64(low, high)};
    const float64x2_t scaled {vaddq_f64(head, vfmaq_f64(tail, vaddq_f64(head, tail), p))};
    const int64x2_t   e      {vshrq_n_s64(n, 7)};
    // e / 2 toward zero, as the scalar division: an odd negative e's half rounded up
    const int64x2_t   half   {vshrq_n_s64(vaddq_s64(e, vreinterpretq_s64_u64(vshrq_n_u64(vreinterpretq_u64_s64(e), 63))), 1)};
    const int64x2_t   bias   {vdupq_n_s64(1023)};
    const float64x2_t first  {vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(half, bias), 52))};
    const float64x2_t second {vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(vsubq_s64(e, half), bias), 52))};
    const float64x2_t result {vmulq_f64(vmulq_f64(scaled, first), second)};
    return vbslq_f64(vceqq_f64(x, x), result, x);
  }

  //! \brief \ref two_sum of each lane.
  NUMERICAL_ALWAYS_INLINE void two_sum(float64x2_t  a,
                                       float64x2_t  b,
                                       float64x2_t& sum,
                                       float64x2_t& error) noexcept
  {
    sum = vaddq_f64(a, b);
    const float64x2_t b_part {vsubq_f64(sum, a)};
    error = vaddq_f64(vsubq_f64(a, vsubq_f64(sum, b_part)), vsubq_f64(b, b_part));
  }

  //! \brief \ref table_expm1 of each lane, in NEON: each operation the scalar one's, so each lane the scalar call's bits.
  NUMERICAL_ALWAYS_INLINE float64x2_t table_expm1(float64x2_t x) noexcept
  {
    const float64x2_t shifter {vdupq_n_f64(0x1.8p52)};
    // NaN stays NaN through both: vminq_f64 and vmaxq_f64 return it
    const float64x2_t clamped {vmaxq_f64(vminq_f64(x, vdupq_n_f64(710.0)), vdupq_n_f64(-40.0))};
    const float64x2_t z       {vfmaq_f64(shifter, clamped, vdupq_n_f64(exp_inv_ln2_128))};
    const int64x2_t   n       {vsubq_s64(vreinterpretq_s64_f64(z), vreinterpretq_s64_f64(shifter))};
    const float64x2_t k       {vsubq_f64(z, shifter)};
    const float64x2_t r_head  {vfmsq_f64(clamped, k, vdupq_n_f64(exp_ln2_128_high))};
    const float64x2_t r       {vfmsq_f64(r_head, k, vdupq_n_f64(exp_ln2_128_low))};
    float64x2_t       q       {vfmaq_f64(vdupq_n_f64(1.0 / 720.0), r, vdupq_n_f64(1.0 / 5040.0))};
    q = vfmaq_f64(vdupq_n_f64(1.0 / 120.0), r, q);
    q = vfmaq_f64(vdupq_n_f64(1.0 / 24.0), r, q);
    q = vfmaq_f64(vdupq_n_f64(1.0 / 6.0), r, q);
    q = vfmaq_f64(vdupq_n_f64(0.5), r, q);
    const float64x2_t square   {vmulq_f64(r, r)};
    const float64x2_t rest     {vmulq_f64(square, q)};
    const float64x2_t p        {vfmaq_f64(r, square, q)};
    const float64x2_t r_rest   {vfmsq_f64(rest, k, vdupq_n_f64(exp_ln2_128_low))};
    const int64x2_t   j        {vandq_s64(n, vdupq_n_s64(127))};
    const float64x2_t low      {vld1q_f64(exp_table.data() + (2 * static_cast<std::size_t>(vgetq_lane_s64(j, 0))))};
    const float64x2_t high     {vld1q_f64(exp_table.data() + (2 * static_cast<std::size_t>(vgetq_lane_s64(j, 1))))};
    const float64x2_t head     {vzip1q_f64(low, high)};
    const float64x2_t tail     {vzip2q_f64(low, high)};
    const int64x2_t   e        {vshrq_n_s64(n, 7)};
    const int64x2_t   cap      {vdupq_n_s64(1022)};
    const int64x2_t   capped   {vbslq_s64(vcgtq_s64(e, cap), cap, e)}; // NEON has no 64-bit minimum
    const float64x2_t unit     {vreinterpretq_f64_s64(vshlq_n_s64(vsubq_s64(vdupq_n_s64(1023), capped), 52))};
    float64x2_t       lead     {};
    float64x2_t       lead_err {};
    two_sum(head, vnegq_f64(unit), lead, lead_err);
    const float64x2_t product  {vmulq_f64(head, r_head)};
    const float64x2_t prod_err {vfmaq_f64(vnegq_f64(product), head, r_head)};
    float64x2_t       sum      {};
    float64x2_t       sum_err  {};
    two_sum(lead, product, sum, sum_err);
    const float64x2_t small    {vaddq_f64(vaddq_f64(vaddq_f64(lead_err, sum_err), prod_err),
                                          vfmaq_f64(vfmaq_f64(tail, tail, p), head, r_rest))};
    const int64x2_t   half     {vshrq_n_s64(vaddq_s64(e, vreinterpretq_s64_u64(vshrq_n_u64(vreinterpretq_u64_s64(e), 63))), 1)};
    const int64x2_t   bias     {vdupq_n_s64(1023)};
    const float64x2_t first    {vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(half, bias), 52))};
    const float64x2_t second   {vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(vsubq_s64(e, half), bias), 52))};
    const float64x2_t result   {vmulq_f64(vmulq_f64(vaddq_f64(sum, small), first), second)};
    return vbslq_f64(vcgeq_f64(vabsq_f64(x), vdupq_n_f64(0x1p-54)), result, x);
  }

#endif
} // namespace numerical::detail

#endif // NUMERICAL_ELEMENTARY_HPP
