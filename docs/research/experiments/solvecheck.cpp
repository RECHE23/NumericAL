// Validation de bout en bout de la substitution par blocs dans une copie modifiée des en-têtes : bits, résidu, temps.
#include <numerical/numerical.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
using numerical::matrix;
int main() {
  if (const char* t = std::getenv("NT")) numerical::set_max_threads(std::strtoul(t, nullptr, 10));
  for (std::size_t n : {64, 128, 256, 512, 1024, 2048}) {
    std::mt19937_64 g(n); std::normal_distribution<double> d;
    std::vector<double> va(n * n), vb(n * 4);
    for (auto& x : va) x = d(g); for (auto& x : vb) x = d(g);
    const matrix<double> a {n, n, std::vector<double>(va)}, b {n, 4, std::vector<double>(vb)};
    auto run = [&](const char* mode) {
      unsetenv("NARROW_OLD"); unsetenv("NARROW_FWD_ONLY");
      if (mode) setenv(mode, "1", 1);
      matrix<double> x {numerical::solve(a, b)};
      double best = 1e9;
      const int reps = std::max<int>(3, int(3e8 / double(n * n * n)));
      for (int r = 0; r < reps; ++r) { auto t0 = std::chrono::steady_clock::now(); x = numerical::solve(a, b); best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
      return std::make_pair(x, best);
    };
    auto [xo, to] = run("NARROW_OLD");
    auto [xf, tf] = run("NARROW_FWD_ONLY");
    auto [xn, tn] = run(nullptr);
    bool same_fwd = true; double diff = 0, scale = 0;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t c = 0; c < 4; ++c) {
      same_fwd &= xo(i, c) == xf(i, c);
      diff = std::max(diff, std::fabs(xo(i, c) - xn(i, c))); scale = std::max(scale, std::fabs(xo(i, c)));
    }
    std::printf("n=%5zu  actuel %8.3f ms | avant par blocs %8.3f ms (mêmes bits : %s) | avant+arrière %8.3f ms (écart rel. %.1e) → %.2f×\n",
                n, to, tf, same_fwd ? "oui" : "NON", tn, diff / scale, to / tn);
  }
}
