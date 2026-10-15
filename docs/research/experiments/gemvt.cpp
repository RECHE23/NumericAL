// BLAS 2.5 : w = A z et x = Aᵀ y en un seul passage sur A (Howell et al., ACM TOMS 2008), contre deux passages.
// Mêmes sommes, dans le même ordre, pour chaque élément : les deux variantes donnent les mêmes bits.
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using clk = std::chrono::steady_clock;

__attribute__((target("avx512f"))) double dot(const double* a, const double* b, std::size_t n) {
  __m512d s0 = _mm512_setzero_pd(), s1 = s0, s2 = s0, s3 = s0; std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), s0);
    s1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 8), _mm512_loadu_pd(b + i + 8), s1);
    s2 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 16), _mm512_loadu_pd(b + i + 16), s2);
    s3 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 24), _mm512_loadu_pd(b + i + 24), s3);
  }
  double s = _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3)));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
__attribute__((target("avx512f"))) void axpy(double* x, const double* a, double f, std::size_t n) {
  const __m512d ff = _mm512_set1_pd(f); std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm512_storeu_pd(x + i, _mm512_fmadd_pd(ff, _mm512_loadu_pd(a + i), _mm512_loadu_pd(x + i)));
  for (; i < n; ++i) x[i] = std::fma(f, a[i], x[i]);
}
int main() {
  for (std::size_t n : {512, 1024, 2048, 4096}) {
    std::mt19937_64 g(n); std::uniform_real_distribution<double> d(-1, 1);
    std::vector<double> a(n * n), z(n), y(n), w1(n), w2(n), x1(n), x2(n);
    for (auto& v : a) v = d(g); for (auto& v : z) v = d(g); for (auto& v : y) v = d(g);
    auto two = [&] { for (std::size_t i = 0; i < n; ++i) w1[i] = dot(a.data() + i * n, z.data(), n);
                     std::fill(x1.begin(), x1.end(), 0.0); for (std::size_t i = 0; i < n; ++i) axpy(x1.data(), a.data() + i * n, y[i], n); };
    auto fused = [&] { std::fill(x2.begin(), x2.end(), 0.0);
                       for (std::size_t i = 0; i < n; ++i) { w2[i] = dot(a.data() + i * n, z.data(), n); axpy(x2.data(), a.data() + i * n, y[i], n); } };
    auto best = [&](auto f) { double b = 1e9; for (int r = 0; r < 9; ++r) { auto t0 = clk::now(); f(); b = std::min(b, std::chrono::duration<double, std::milli>(clk::now() - t0).count()); } return b; };
    const double t2 = best(two), t1 = best(fused);
    const bool same = !std::memcmp(w1.data(), w2.data(), n * 8) && !std::memcmp(x1.data(), x2.data(), n * 8);
    std::printf("n=%4zu (%5.1f Mo) : deux passages %7.3f ms, un passage %7.3f ms → %.2f×, mêmes bits : %s\n", n, n * n * 8 / 1e6, t2, t1, t2 / t1, same ? "oui" : "NON");
  }
}
