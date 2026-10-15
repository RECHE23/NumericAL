// Débit du noyau AMX-INT8 seul (un module, sans épilogue) selon le blocage, sur un fil.
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
struct cfg_t { uint8_t palette{1}, start{0}, res[14]{}; uint16_t colsb[16]{}; uint8_t rows[16]{}; };
int main() {
  syscall(SYS_arch_prctl, 0x1023, 18);
  for (std::size_t n : {512, 1024, 2048}) {
    const std::size_t kt = n / 64, mt = n / 16, nt = n / 16;
    std::vector<int8_t> a(mt * kt * 1024, 1), b(nt * kt * 1024, 1);
    std::vector<int32_t> c(n * n);
    alignas(64) cfg_t cfg; for (int t = 0; t < 8; ++t) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
    _tile_loadconfig(&cfg);
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      for (std::size_t ib = 0; ib < mt / 2; ++ib) {
        const int8_t* a0 = a.data() + (2 * ib) * kt * 1024; const int8_t* a1 = a0 + kt * 1024;
        for (std::size_t jb = 0; jb < nt / 2; ++jb) {
          const int8_t* b0 = b.data() + (2 * jb) * kt * 1024; const int8_t* b1 = b0 + kt * 1024;
          _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
          for (std::size_t k = 0; k < kt; ++k) {
            _tile_loadd(4, a0 + k * 1024, 64); _tile_loadd(6, b0 + k * 1024, 64);
            _tile_loadd(5, a1 + k * 1024, 64); _tile_loadd(7, b1 + k * 1024, 64);
            _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7);
          }
          int32_t* cc = c.data() + ib * 32 * n + jb * 32;
          _tile_stored(0, cc, n * 4); _tile_stored(1, cc + 16, n * 4); _tile_stored(2, cc + 16 * n, n * 4); _tile_stored(3, cc + 16 * n + 16, n * 4);
        }
      }
      best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("n=%zu : %.2f ms, %.2f TMAC/s (%.0f %% de la crête mesurée 2,5)\n", n, best * 1e3, double(n) * n * n / best / 1e12, double(n) * n * n / best / 1e12 / 2.5 * 100);
    _tile_release();
  }
}
