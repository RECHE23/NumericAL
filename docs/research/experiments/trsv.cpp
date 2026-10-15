// Substitution avant L·Y = B (L unitaire triangulaire inférieure, Y n × c, c ≤ 8) :
// la boucle actuelle de NumericAL (une ligne à la fois) contre un bloc de 8 lignes dont les chaînes de FMA
// sont indépendantes. Chaque élément reçoit ses termes dans le même ordre (j croissant) : mêmes bits attendus.
#include <numerical/numerical.hpp>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

// Référence : la boucle de solve_narrow (passe avant).
void forward_current(std::vector<double>& y, const std::vector<double>& lu, std::size_t n, std::size_t cols)
{
  for (std::size_t i {1}; i < n; ++i) {
    numerical::detail::subtract_products(y.data() + (i * cols), lu.data() + (i * n), y.data(), cols, i, cols);
  }
}

// Proposition : blocs de R lignes ; les termes j < i0 appliqués aux R lignes ensemble (R chaînes indépendantes),
// puis la partie triangulaire du bloc ligne par ligne. cols == 4 : une ligne de Y tient dans un registre de 4 doubles.
template <std::size_t R>
__attribute__((target("avx512f,avx512vl,fma"))) void forward_blocked4(std::vector<double>& y, const std::vector<double>& lu, std::size_t n)
{
  constexpr std::size_t cols {4};
  std::size_t i0 {0};
  for (; i0 + R <= n; i0 += R) {
    __m256d acc[R];
    for (std::size_t r {0}; r < R; ++r) acc[r] = _mm256_loadu_pd(y.data() + ((i0 + r) * cols));
    const double* rows[R];
    for (std::size_t r {0}; r < R; ++r) rows[r] = lu.data() + ((i0 + r) * n);
    for (std::size_t j {0}; j < i0; ++j) {
      const __m256d xj {_mm256_loadu_pd(y.data() + (j * cols))};
#pragma GCC unroll 16
      for (std::size_t r {0}; r < R; ++r) acc[r] = _mm256_fnmadd_pd(_mm256_set1_pd(rows[r][j]), xj, acc[r]);
    }
    // La partie triangulaire du bloc : la ligne r reçoit les termes j = i0 … i0 + r − 1, dans l'ordre.
    for (std::size_t r {0}; r < R; ++r) {
      for (std::size_t q {0}; q < r; ++q) {
        acc[r] = _mm256_fnmadd_pd(_mm256_set1_pd(rows[r][i0 + q]), acc[q], acc[r]);
      }
      _mm256_storeu_pd(y.data() + ((i0 + r) * cols), acc[r]);
    }
  }
  for (std::size_t i {std::max<std::size_t>(i0, 1)}; i < n; ++i) { // les lignes restantes, comme avant
    numerical::detail::subtract_products(y.data() + (i * cols), lu.data() + (i * n), y.data(), cols, i, cols);
  }
}

int main()
{
  for (std::size_t n : {64, 128, 256, 512, 1024, 2048}) {
    std::mt19937_64 g {n};
    std::uniform_real_distribution<double> d(-1, 1);
    std::vector<double> lu(n * n), b(n * 4);
    for (auto& x : lu) x = d(g) / double(n);
    for (auto& x : b) x = d(g);
    auto y1 = b, y2 = b;
    forward_current(y1, lu, n, 4);
    forward_blocked4<8>(y2, lu, n);
    const bool same = std::memcmp(y1.data(), y2.data(), y1.size() * sizeof(double)) == 0;
    auto bench = [&](auto f) {
      double best {1e9};
      for (int rep = 0; rep < 7; ++rep) {
        auto y = b;
        const int inner = std::max<int>(1, int(2e7 / (n * n)));
        auto t0 = clk::now();
        for (int k = 0; k < inner; ++k) { y = b; f(y); }
        best = std::min(best, std::chrono::duration<double, std::micro>(clk::now() - t0).count() / inner);
      }
      return best;
    };
    const double tc = bench([&](auto& y) { forward_current(y, lu, n, 4); });
    const double tb = bench([&](auto& y) { forward_blocked4<8>(y, lu, n); });
    std::printf("n=%5zu  actuel %9.1f µs  par blocs de 8 %8.1f µs  accélération %4.1f×  mêmes bits : %s\n",
                n, tc, tb, tc / tb, same ? "oui" : "NON");
  }
}
