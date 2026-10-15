// Débits crêtes d'un cœur : FMA FP64 AVX-512 et AMX INT8/BF16 (tuiles en registre, sans mémoire).
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__attribute__((target("avx512f"))) double fma_peak(long iters)
{
  __m512d acc[24];
  for (int i = 0; i < 24; ++i) acc[i] = _mm512_set1_pd(i * 1e-9);
  const __m512d a = _mm512_set1_pd(1.0000001), b = _mm512_set1_pd(0.9999999);
  for (long it = 0; it < iters; ++it) {
#pragma GCC unroll 24
    for (int i = 0; i < 24; ++i) acc[i] = _mm512_fmadd_pd(acc[i], a, b);
  }
  double s = 0;
  for (int i = 0; i < 24; ++i) s += _mm512_reduce_add_pd(acc[i]);
  return s;
}

struct tilecfg { uint8_t palette, start; uint8_t pad[14]; uint16_t colsb[16]; uint8_t rows[16]; };

__attribute__((target("amx-tile,amx-int8,amx-bf16"))) void amx_peak(long iters, bool bf16)
{
  alignas(64) tilecfg cfg {};
  cfg.palette = 1;
  for (int t = 0; t < 8; ++t) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
  _tile_loadconfig(&cfg);
  alignas(64) static int8_t buf[16 * 64] {};
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  _tile_loadd(4, buf, 64); _tile_loadd(5, buf, 64); _tile_loadd(6, buf, 64); _tile_loadd(7, buf, 64);
  if (bf16) {
    for (long it = 0; it < iters; ++it) {
      _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
    }
  } else {
    for (long it = 0; it < iters; ++it) {
      _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7);
    }
  }
  alignas(64) static int32_t out[16 * 16];
  _tile_stored(0, out, 64);
  _tile_release();
}

int main()
{
  long it = 20000000;
  double t0 = now(); volatile double s = fma_peak(it); double t1 = now();
  std::printf("FP64 AVX-512 FMA : %.1f GFLOP/s par coeur (%g)\n", it * 24.0 * 16 / (t1 - t0) / 1e9, (double)s);
  if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) { std::printf("AMX refusé par le noyau\n"); return 0; }
  long ia = 2000000;
  t0 = now(); amx_peak(ia, false); t1 = now();
  std::printf("AMX INT8 (TDPBSSD) : %.1f TOPS par coeur (%.0f MAC/cycle à 2,1 GHz nominal)\n",
              ia * 4.0 * 16 * 16 * 64 * 2 / (t1 - t0) / 1e12, ia * 4.0 * 16 * 16 * 64 / (t1 - t0) / 2.1e9);
  t0 = now(); amx_peak(ia, true); t1 = now();
  std::printf("AMX BF16 (TDPBF16PS) : %.1f TFLOP/s par coeur\n", ia * 4.0 * 16 * 16 * 32 * 2 / (t1 - t0) / 1e12);
}
