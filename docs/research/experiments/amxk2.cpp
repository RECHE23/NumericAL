// Variantes du noyau AMX-INT8 : opérandes résidents en L1, ordre des chargements entrelacé, blocage en profondeur.
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
struct cfg_t { uint8_t palette{1}, start{0}, res[14]{}; uint16_t colsb[16]{}; uint8_t rows[16]{}; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  syscall(SYS_arch_prctl, 0x1023, 18);
  alignas(64) cfg_t cfg; for (int t = 0; t < 8; ++t) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
  _tile_loadconfig(&cfg);
  alignas(64) static int8_t l1[4][1024] {};
  const long it = 2000000;
  double t0 = now();
  for (long k = 0; k < it; ++k) {
    _tile_loadd(4, l1[0], 64); _tile_loadd(6, l1[2], 64); _tile_loadd(5, l1[1], 64); _tile_loadd(7, l1[3], 64);
    _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7);
  }
  double t = now() - t0;
  std::printf("2×2, chargements depuis le L1 : %.2f TMAC/s\n", it * 4.0 * 16384 / t / 1e12);
  t0 = now();
  for (long k = 0; k < it; ++k) { // chargements placés dès que la tuile est libre
    _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7); _tile_loadd(4, l1[0], 64);
    _tile_dpbssd(2, 5, 6); _tile_loadd(6, l1[2], 64);
    _tile_dpbssd(3, 5, 7); _tile_loadd(5, l1[1], 64); _tile_loadd(7, l1[3], 64);
  }
  t = now() - t0;
  std::printf("2×2, chargements entrelacés (L1) : %.2f TMAC/s\n", it * 4.0 * 16384 / t / 1e12);
  // Profondeur bloquée : A et B (32 × kc et kc × 32) dans le L1, réutilisés.
  for (std::size_t kc : {128, 256, 384}) {
    std::vector<int8_t> a(2 * kc / 64 * 1024, 1), b(2 * kc / 64 * 1024, 1);
    const long reps = 4000000 / (kc / 64);
    t0 = now();
    for (long r = 0; r < reps; ++r) {
      for (std::size_t k = 0; k < kc / 64; ++k) {
        _tile_loadd(4, a.data() + k * 1024, 64); _tile_loadd(6, b.data() + k * 1024, 64);
        _tile_loadd(5, a.data() + (kc / 64 + k) * 1024, 64); _tile_loadd(7, b.data() + (kc / 64 + k) * 1024, 64);
        _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7);
      }
    }
    t = now() - t0;
    std::printf("2×2, kc = %zu en L1 : %.2f TMAC/s\n", kc, reps * double(kc / 64) * 4 * 16384 / t / 1e12);
  }
  _tile_release();
}
