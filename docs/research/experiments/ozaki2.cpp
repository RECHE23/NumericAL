// Prototype : DGEMM émulée par le schéma d'Ozaki II (restes chinois) sur les unités AMX-INT8 d'un Xeon.
//
// C = A·B (double) se calcule ainsi :
//   1. mise à l'échelle : la ligne i de A par 2^μᵢ, la colonne j de B par 2^νⱼ, puis troncature en entiers
//      Ā, B̄ (exacte : un double tronqué reste un double), de sorte que |Ā·B̄| < P/4 (Cauchy-Schwarz) ;
//   2. pour chaque module pₗ (deux à deux premiers entre eux, ≤ 256) : les résidus symétriques Āₗ, B̄ₗ en int8 ;
//   3. Cₗ = Āₗ·B̄ₗ en int32, exact, sur l'AMX (TDPBSSD), réduit modulo pₗ dans l'épilogue en octets rₗ ∈ [0, pₗ) ;
//   4. reconstruction : S = Σ rₗ·Q'ₗ (Q'ₗ = (P/pₗ)·((P/pₗ)⁻¹ mod pₗ) mod P) en fragments de 40 bits, exacts en
//      double ; C̄ = S − P·round(S/P) ; puis Cᵢⱼ = C̄ᵢⱼ·2^−(μᵢ+νⱼ).
// Toutes les étapes entières sont exactes : le résultat ne dépend ni de l'ordre des sommes ni du nombre de fils.
//
// Compilation : g++ -std=c++20 -O3 -march=sapphirerapids -I<NumericAL>/include -pthread ozaki2.cpp
#include <numerical/numerical.hpp>

#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

  using clk = std::chrono::steady_clock;
  double since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

  constexpr int max_moduli {16};
  constexpr int chunks     {4};    // fragments de 40 bits : jusqu'à 160 bits, assez pour Σ rₗ Q'ₗ < N·256·P
  constexpr int moduli_list[max_moduli] {256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193};

  using u128 = unsigned __int128;

  long modinv(long a, long m) // inverse de a modulo m (a, m premiers entre eux), par Euclide étendu
  {
    long g {m}, x {0}, x1 {1}, a1 {a % m};
    while (a1 != 0) {
      const long q {g / a1};
      std::swap(g, a1);  a1 -= q * g;
      std::swap(x, x1);  x1 -= q * x;
    }
    return ((x % m) + m) % m;
  }

  struct crt_plan
  {
    int    n_moduli {};
    int    beta {};                         // bits par côté : |Āᵢ|₂ < 2^β, |B̄ⱼ|₂ < 2^β, |C̄| < 2^2β ≤ P/4
    float  p_f[max_moduli] {}, inv_p_f[max_moduli] {};
    double p[max_moduli] {}, inv_p[max_moduli] {}, half_hi[max_moduli] {}, half_lo[max_moduli] {};
    double q_chunk[max_moduli][chunks] {};  // Q'ₗ en fragments de 40 bits
    double p_chunk[chunks] {};              // P en fragments de 40 bits
    double chunk_over_p[chunks] {};         // 2^(40c)/P, pour round(S/P)

    explicit crt_plan(int n) : n_moduli {n}
    {
      u128 big {1};
      for (int l {0}; l < n; ++l) { big *= static_cast<u128>(moduli_list[l]); }
      const double log2p {std::log2(static_cast<double>(big))};
      beta = static_cast<int>(std::floor((log2p - 2.0) / 2.0));
      for (int c {0}; c < chunks; ++c) {
        p_chunk[c]      = static_cast<double>(static_cast<std::uint64_t>((big >> (40 * c)) & ((u128 {1} << 40) - 1)));
        chunk_over_p[c] = std::ldexp(1.0, 40 * c) / static_cast<double>(big);
      }
      for (int l {0}; l < n; ++l) {
        const long pl {moduli_list[l]};
        const u128 cofactor {big / static_cast<u128>(pl)};
        const long y {modinv(static_cast<long>(cofactor % static_cast<u128>(pl)), pl)};
        const u128 q {(cofactor * static_cast<u128>(y)) % big};
        for (int c {0}; c < chunks; ++c) {
          q_chunk[l][c] = static_cast<double>(static_cast<std::uint64_t>((q >> (40 * c)) & ((u128 {1} << 40) - 1)));
        }
        p[l]       = static_cast<double>(pl);
        inv_p[l]   = 1.0 / p[l];
        p_f[l]     = static_cast<float>(pl);
        inv_p_f[l] = 1.0f / p_f[l];
        half_hi[l] = static_cast<double>((pl - 1) / 2);                 // résidu symétrique : [−⌊p/2⌋, ⌊(p−1)/2⌋]
        half_lo[l] = -static_cast<double>(pl / 2);
      }
    }
  };

  // L'exposant e tel que la norme 2 du vecteur x (stride s) soit < 2^e ; un vecteur nul donne une échelle neutre.
  int norm_exponent(const double* x, std::size_t count, std::size_t stride)
  {
    double largest {0.0};
    for (std::size_t i {0}; i < count; ++i) { largest = std::max(largest, std::fabs(x[i * stride])); }
    if (largest == 0.0) { return 0; }
    int e_max {};
    std::frexp(largest, &e_max);
    const double down {std::ldexp(1.0, -e_max)};
    double       sum {0.0};
    for (std::size_t i {0}; i < count; ++i) {
      const double v {x[i * stride] * down};
      sum += v * v;
    }
    int e {};
    std::frexp(std::sqrt(sum) * (1.0 + 1e-12), &e);
    return e + e_max;
  }

  // Résidus symétriques de 16 entiers (deux registres de doubles) modulo p, en 16 octets signés.
  __m128i residues16(__m512d x0, __m512d x1, const crt_plan& plan, int l)
  {
    const __m512d p {_mm512_set1_pd(plan.p[l])}, ip {_mm512_set1_pd(plan.inv_p[l])};
    const __m512d hi {_mm512_set1_pd(plan.half_hi[l])}, lo {_mm512_set1_pd(plan.half_lo[l])};
    auto one = [&](__m512d x) {
      // Deux réductions : jusqu'à 2^63, le quotient arrondi peut se tromper de quelques unités, et le premier reste
      // (exact, la valeur exacte étant un petit entier) dépasser ±p ; le second le ramène dans [−p/2, p/2].
      const __m512d q1 {_mm512_roundscale_pd(_mm512_mul_pd(x, ip), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
      const __m512d r1 {_mm512_fnmadd_pd(q1, p, x)};
      const __m512d q {_mm512_roundscale_pd(_mm512_mul_pd(r1, ip), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
      __m512d       r {_mm512_fnmadd_pd(q, p, r1)};
      r = _mm512_mask_sub_pd(r, _mm512_cmp_pd_mask(r, hi, _CMP_GT_OQ), r, p);
      r = _mm512_mask_add_pd(r, _mm512_cmp_pd_mask(r, lo, _CMP_LT_OQ), r, p);
      return _mm512_cvtpd_epi32(r);
    };
    const __m512i both {_mm512_inserti64x4(_mm512_castsi256_si512(one(x0)), one(x1), 1)};
    return _mm512_cvtepi32_epi8(both);
  }

  struct packed_operands
  {
    std::size_t          m {}, n {}, k {}, mt {}, nt {}, kt {};   // tuiles de 16 lignes, 16 colonnes, 64 de profondeur
    std::vector<int8_t>  a, b;                                     // [module][tuile] de 1 Kio chacune
    std::vector<int>     mu, nu;
    std::vector<double>  row_up, col_up, row_down, col_down;      // 2^μᵢ, 2^νⱼ et leurs inverses (exacts)
    std::size_t a_tiles() const { return mt * kt; }
    std::size_t b_tiles() const { return nt * kt; }
  };

  // Étapes 1 et 2 : mise à l'échelle, troncature, résidus, rangement en tuiles AMX (A ligne par ligne, B en VNNI).
  // Les tampons sont gardés d'un appel à l'autre (comme un espace de travail) : ni allocation ni mise à zéro, toutes
  // les tuiles étant réécrites en entier ; les lignes de remplissage, jamais écrites, restent nulles.
  void prepare(packed_operands& o, const double* a, const double* b, std::size_t m, std::size_t n, std::size_t k, const crt_plan& plan)
  {
    o.m  = m; o.n = n; o.k = k;
    o.mt = (m + 31) / 32 * 2; o.nt = (n + 31) / 32 * 2; o.kt = (k + 63) / 64;
    const int N {plan.n_moduli};
    o.a.resize(static_cast<std::size_t>(N) * o.a_tiles() * 1024, 0);
    o.b.resize(static_cast<std::size_t>(N) * o.b_tiles() * 1024, 0);
    o.mu.resize(o.mt * 16, 0);
    o.nu.resize(o.nt * 16, 0);
    o.row_up.resize(o.mt * 16, 1.0); o.row_down.resize(o.mt * 16, 1.0);
    o.col_up.resize(o.nt * 16, 1.0); o.col_down.resize(o.nt * 16, 1.0);
    numerical::detail::parallel_for(m, k * N * 4, [&](std::size_t i) {
      o.mu[i]       = plan.beta - norm_exponent(a + (i * k), k, 1);
      o.row_up[i]   = std::ldexp(1.0, o.mu[i]);
      o.row_down[i] = std::ldexp(1.0, -o.mu[i]);
      const double up {o.row_up[i]};
      alignas(64) double row[64];
      for (std::size_t kt {0}; kt < o.kt; ++kt) {
        for (std::size_t c {0}; c < 64; ++c) {
          const std::size_t col {(kt * 64) + c};
          row[c] = col < k ? std::trunc(a[(i * k) + col] * up) : 0.0;
        }
        for (int l {0}; l < N; ++l) {
          int8_t* tile {o.a.data() + ((static_cast<std::size_t>(l) * o.a_tiles() + ((i / 16) * o.kt) + kt) * 1024) + ((i % 16) * 64)};
          for (std::size_t c {0}; c < 64; c += 16) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tile + c), residues16(_mm512_load_pd(row + c), _mm512_load_pd(row + c + 8), plan, l));
          }
        }
      }
    });
    numerical::detail::parallel_for(o.nt, k * 16 * 2, [&](std::size_t jt) {
      double largest[16] {}, sum[16] {};
      const std::size_t width {std::min<std::size_t>(16, n - std::min(n, jt * 16))};
      for (std::size_t h {0}; h < k; ++h) {
        for (std::size_t c {0}; c < width; ++c) { largest[c] = std::max(largest[c], std::fabs(b[(h * n) + (jt * 16) + c])); }
      }
      double down[16];
      int    e_max[16];
      for (std::size_t c {0}; c < 16; ++c) {
        std::frexp(largest[c], &e_max[c]);
        down[c] = std::ldexp(1.0, -e_max[c]);
      }
      for (std::size_t h {0}; h < k; ++h) {
        for (std::size_t c {0}; c < width; ++c) {
          const double v {b[(h * n) + (jt * 16) + c] * down[c]};
          sum[c] += v * v;
        }
      }
      for (std::size_t c {0}; c < width; ++c) {
        const std::size_t j {(jt * 16) + c};
        int               e {};
        std::frexp(std::sqrt(sum[c]) * (1.0 + 1e-12), &e);
        o.nu[j]       = largest[c] == 0.0 ? 0 : plan.beta - (e + e_max[c]);
        o.col_up[j]   = std::ldexp(1.0, o.nu[j]);
        o.col_down[j] = std::ldexp(1.0, -o.nu[j]);
      }
    });
    // B : par groupes de 4 lignes de profondeur et 16 colonnes, entrelacés octet par octet (format VNNI).
    numerical::detail::parallel_for(o.nt, k * 16 * N * 4, [&](std::size_t jt) {
      alignas(64) double block[4][16];
      for (std::size_t k4 {0}; k4 < o.kt * 16; ++k4) { // une ligne de tuile : 4 profondeurs × 16 colonnes
        for (int q {0}; q < 4; ++q) {
          const std::size_t row {(k4 * 4) + q};
          for (int c {0}; c < 16; ++c) {
            const std::size_t col {(jt * 16) + c};
            block[q][c] = (row < k && col < n) ? std::trunc(b[(row * n) + col] * o.col_up[col]) : 0.0;
          }
        }
        for (int l {0}; l < N; ++l) {
          __m128i r[4];
          for (int q {0}; q < 4; ++q) { r[q] = residues16(_mm512_load_pd(block[q]), _mm512_load_pd(block[q] + 8), plan, l); }
          const __m128i t01lo {_mm_unpacklo_epi8(r[0], r[1])}, t01hi {_mm_unpackhi_epi8(r[0], r[1])};
          const __m128i t23lo {_mm_unpacklo_epi8(r[2], r[3])}, t23hi {_mm_unpackhi_epi8(r[2], r[3])};
          int8_t* tile {o.b.data() + ((static_cast<std::size_t>(l) * o.b_tiles() + (jt * o.kt) + (k4 / 16)) * 1024) + ((k4 % 16) * 64)};
          _mm_storeu_si128(reinterpret_cast<__m128i*>(tile),      _mm_unpacklo_epi16(t01lo, t23lo));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(tile + 16), _mm_unpackhi_epi16(t01lo, t23lo));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(tile + 32), _mm_unpacklo_epi16(t01hi, t23hi));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(tile + 48), _mm_unpackhi_epi16(t01hi, t23hi));
        }
      }
    });
  }

  struct tile_config
  {
    std::uint8_t  palette {1}, start {0}, reserved[14] {};
    std::uint16_t colsb[16] {};
    std::uint8_t  rows[16] {};
  };

  void configure_tiles()
  {
    alignas(64) tile_config cfg {};
    for (int t {0}; t < 8; ++t) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
    _tile_loadconfig(&cfg);
  }

  // Étape 3 : pour chaque module, Cₗ = Āₗ·B̄ₗ par blocs 32 × 32 (quatre tuiles d'accumulateurs), réduit modulo pₗ.
  // Les résidus vont dans des plans d'octets : r[l][i][j]. Les tâches d'un même module se suivent : B̄ₗ reste au L2.
  void products(const packed_operands& o, const crt_plan& plan, std::vector<std::uint8_t>& r)
  {
    const std::size_t ldr {o.nt * 16}, plane {o.mt * 16 * ldr};
    r.resize(static_cast<std::size_t>(plan.n_moduli) * plane);
    const std::size_t row_blocks {o.mt / 2};
    const bool        deep {o.kt * 64 > 1024};
    numerical::detail::parallel_for(row_blocks * plan.n_moduli, 32 * o.nt * 16 * o.kt * 64 / 64, [&](std::size_t task) {
      const int         l  {static_cast<int>(task / row_blocks)};
      const std::size_t ib {task % row_blocks};
      configure_tiles();
      alignas(64) std::int32_t c[32][32];
      const int8_t* a0 {o.a.data() + ((static_cast<std::size_t>(l) * o.a_tiles() + ((2 * ib) * o.kt)) * 1024)};
      const int8_t* a1 {a0 + (o.kt * 1024)};
      const __m512 p {_mm512_set1_ps(plan.p_f[l])}, ip {_mm512_set1_ps(plan.inv_p_f[l])}, zero {_mm512_setzero_ps()};
      for (std::size_t jb {0}; jb < o.nt / 2; ++jb) {
        const int8_t* b0 {o.b.data() + ((static_cast<std::size_t>(l) * o.b_tiles() + ((2 * jb) * o.kt)) * 1024)};
        const int8_t* b1 {b0 + (o.kt * 1024)};
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
        for (std::size_t kt {0}; kt < o.kt; ++kt) {
          _tile_loadd(4, a0 + (kt * 1024), 64);
          _tile_loadd(6, b0 + (kt * 1024), 64);
          _tile_loadd(5, a1 + (kt * 1024), 64);
          _tile_loadd(7, b1 + (kt * 1024), 64);
          _tile_dpbssd(0, 4, 6);
          _tile_dpbssd(1, 4, 7);
          _tile_dpbssd(2, 5, 6);
          _tile_dpbssd(3, 5, 7);
        }
        _tile_stored(0, &c[0][0], 128);
        _tile_stored(1, &c[0][16], 128);
        _tile_stored(2, &c[16][0], 128);
        _tile_stored(3, &c[16][16], 128);
        // Épilogue : c mod p dans [0, p). En simple précision tant que |c| ≤ 2^24 (profondeur ≤ 1024 : exact en
        // float), en double au-delà.
        for (int i {0}; i < 32; ++i) {
          for (int j {0}; j < 32; j += 16) {
            __m512i residue;
            if (deep) {
              const __m512d pd {_mm512_set1_pd(plan.p[l])}, ipd {_mm512_set1_pd(plan.inv_p[l])}, zd {_mm512_setzero_pd()};
              auto half = [&](__m256i x) {
                const __m512d v {_mm512_cvtepi32_pd(x)};
                __m512d       rr {_mm512_fnmadd_pd(_mm512_floor_pd(_mm512_mul_pd(v, ipd)), pd, v)};
                rr = _mm512_mask_add_pd(rr, _mm512_cmp_pd_mask(rr, zd, _CMP_LT_OQ), rr, pd);
                rr = _mm512_mask_sub_pd(rr, _mm512_cmp_pd_mask(rr, pd, _CMP_GE_OQ), rr, pd);
                return _mm512_cvtpd_epi32(rr);
              };
              const __m512i all {_mm512_load_si512(&c[i][j])};
              residue = _mm512_inserti64x4(_mm512_castsi256_si512(half(_mm512_castsi512_si256(all))), half(_mm512_extracti64x4_epi64(all, 1)), 1);
            }
            else {
              const __m512 v {_mm512_cvtepi32_ps(_mm512_load_si512(&c[i][j]))};
              __m512       rr {_mm512_fnmadd_ps(_mm512_floor_ps(_mm512_mul_ps(v, ip)), p, v)};
              rr = _mm512_mask_add_ps(rr, _mm512_cmp_ps_mask(rr, zero, _CMP_LT_OQ), rr, p);
              rr = _mm512_mask_sub_ps(rr, _mm512_cmp_ps_mask(rr, p, _CMP_GE_OQ), rr, p);
              residue = _mm512_cvtps_epi32(rr);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(r.data() + (static_cast<std::size_t>(l) * plane) + (((ib * 32) + i) * ldr) + (jb * 32) + j),
                             _mm512_cvtepi32_epi8(residue));
          }
        }
      }
      _tile_release();
    });
  }

  // Étape 4 : reconstruction par fragments exacts, puis la mise à l'échelle inverse.
  void reconstruct(const std::vector<std::uint8_t>& r, const packed_operands& o, const crt_plan& plan, double* out)
  {
    const std::size_t ldr {o.nt * 16}, plane {o.mt * 16 * ldr};
    numerical::detail::parallel_for(o.m, o.n * plan.n_moduli * 8, [&](std::size_t i) {
      for (std::size_t j {0}; j < o.n; j += 8) {
        __m512d s[chunks];
        for (auto& x : s) { x = _mm512_setzero_pd(); }
        for (int l {0}; l < plan.n_moduli; ++l) {
          const __m128i bytes {_mm_loadl_epi64(reinterpret_cast<const __m128i*>(r.data() + (static_cast<std::size_t>(l) * plane) + (i * ldr) + j))};
          const __m512d v {_mm512_cvtepi32_pd(_mm256_cvtepu8_epi32(bytes))};
          for (int c {0}; c < chunks; ++c) { s[c] = _mm512_fmadd_pd(v, _mm512_set1_pd(plan.q_chunk[l][c]), s[c]); }
        }
        __m512d t {_mm512_setzero_pd()};
        for (int c {chunks}; c-- > 0;) { t = _mm512_fmadd_pd(s[c], _mm512_set1_pd(plan.chunk_over_p[c]), t); }
        const __m512d q {_mm512_roundscale_pd(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
        __m512d       d[chunks];
        for (int c {0}; c < chunks; ++c) { d[c] = _mm512_fnmadd_pd(q, _mm512_set1_pd(plan.p_chunk[c]), s[c]); }
        const __m512d two40 {_mm512_set1_pd(0x1p40)}, inv40 {_mm512_set1_pd(0x1p-40)};
        for (int c {0}; c + 1 < chunks; ++c) { // retenues : chaque fragment ramené à ±2^39
          const __m512d carry {_mm512_roundscale_pd(_mm512_mul_pd(d[c], inv40), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
          d[c]     = _mm512_fnmadd_pd(carry, two40, d[c]);
          d[c + 1] = _mm512_add_pd(d[c + 1], carry);
        }
        __m512d value {d[chunks - 1]};
        for (int c {chunks - 1}; c-- > 0;) { value = _mm512_fmadd_pd(value, two40, d[c]); }
        value = _mm512_mul_pd(_mm512_mul_pd(value, _mm512_set1_pd(o.row_down[i])), _mm512_loadu_pd(o.col_down.data() + j));
        if (j + 8 <= o.n) {
          _mm512_storeu_pd(out + (i * o.n) + j, value);
        }
        else {
          _mm512_mask_storeu_pd(out + (i * o.n) + j, static_cast<__mmask8>((1U << (o.n - j)) - 1U), value);
        }
      }
    });
  }

  // Produit scalaire compensé (Dot2 d'Ogita, Rump et Oishi) : une référence quasi exacte.
  double dot2(const double* a, const double* b, std::size_t k, std::size_t ldb)
  {
    double s {0.0}, c {0.0};
    for (std::size_t h {0}; h < k; ++h) {
      const double p {a[h] * b[h * ldb]};
      const double pe {std::fma(a[h], b[h * ldb], -p)};
      const double t {s + p};
      const double z {t - s};
      c += ((s - (t - z)) + (p - z)) + pe;
      s = t;
    }
    return s + c;
  }

} // namespace

int main(int argc, char** argv)
{
  if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) { std::puts("AMX indisponible"); return 1; }
  const std::size_t n      {argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1024};
  const double      spread {argc > 2 ? std::atof(argv[2]) : 0.0};  // étalement des exposants (φ des articles d'Ozaki)
  const int         reps   {argc > 3 ? std::atoi(argv[3]) : 5};
  if (const char* t = std::getenv("NT")) { numerical::set_max_threads(std::strtoul(t, nullptr, 10)); }
  std::mt19937_64                        g {42};
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  std::normal_distribution<double>       nd;
  std::vector<double>                    a(n * n), b(n * n);
  for (auto& x : a) { x = u(g) * std::exp(spread * nd(g)); }
  for (auto& x : b) { x = u(g) * std::exp(spread * nd(g)); }
  const numerical::matrix<double> ma {n, n, std::vector<double>(a)}, mb {n, n, std::vector<double>(b)};

  // La DGEMM native de NumericAL (chemin AVX-512), pour le temps et l'erreur de référence.
  numerical::matrix<double> native {ma * mb};
  double t_native {1e9};
  for (int r {0}; r < reps; ++r) { const auto t0 {clk::now()}; native = ma * mb; t_native = std::min(t_native, since(t0)); }

  // Échantillon d'éléments pour l'erreur, mesurée relativement à (|A||B|)ᵢⱼ, comme la borne de DGEMM.
  std::vector<std::pair<std::size_t, std::size_t>> sample;
  for (int s {0}; s < 2000; ++s) { sample.emplace_back(g() % n, g() % n); }
  std::vector<double> exact(sample.size()), scale(sample.size());
  for (std::size_t s {0}; s < sample.size(); ++s) {
    const auto [i, j] = sample[s];
    exact[s] = dot2(a.data() + (i * n), b.data() + j, n, n);
    double t {0.0};
    for (std::size_t h {0}; h < n; ++h) { t += std::fabs(a[(i * n) + h] * b[(h * n) + j]); }
    scale[s] = t;
  }
  auto error = [&](auto at) {
    double worst {0.0};
    for (std::size_t s {0}; s < sample.size(); ++s) {
      worst = std::max(worst, std::fabs(at(sample[s].first, sample[s].second) - exact[s]) / scale[s]);
    }
    return worst;
  };
  std::printf("n = %zu, étalement φ = %.1f, %zu fils ; DGEMM native : %.2f ms (%.0f GFLOP/s), erreur %.1e\n", n, spread,
              numerical::max_threads(), t_native, 2.0 * n * n * n / t_native / 1e6,
              error([&](std::size_t i, std::size_t j) { return native(i, j); }));
  std::printf("%8s %10s %10s %10s %10s %10s %9s %10s\n", "modules", "préparer", "produits", "reconstr.", "total", "GFLOP/s", "vs natif", "erreur");
  std::vector<double> out(n * n);
  for (int moduli {12}; moduli <= max_moduli; ++moduli) {
    const crt_plan plan {moduli};
    double tp {1e9}, tg {1e9}, tr {1e9}, tt {1e9};
    std::vector<std::uint8_t> r;
    packed_operands           o;
    for (int rep {0}; rep < reps + 1; ++rep) { // le premier appel alloue l'espace de travail : non compté
      const auto t0 {clk::now()};
      prepare(o, a.data(), b.data(), n, n, n, plan);
      const double a1 {since(t0)};
      products(o, plan, r);
      const double a2 {since(t0)};
      reconstruct(r, o, plan, out.data());
      const double a3 {since(t0)};
      if (rep > 0) { tp = std::min(tp, a1); tg = std::min(tg, a2 - a1); tr = std::min(tr, a3 - a2); tt = std::min(tt, a3); }
    }
    std::printf("%8d %8.2f ms %8.2f ms %8.2f ms %8.2f ms %10.0f %8.2f× %10.1e\n", moduli, tp, tg, tr, tt,
                2.0 * n * n * n / tt / 1e6, t_native / tt, error([&](std::size_t i, std::size_t j) { return out[(i * n) + j]; }));
  }
}
