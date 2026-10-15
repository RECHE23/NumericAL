// Pilote de profilage : une opération de NumericAL répétée, choisie en ligne de commande.
#include <numerical/numerical.hpp>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

using numerical::matrix;

template <typename T> matrix<T> rnd(std::size_t r, std::size_t c, unsigned seed, bool spd = false)
{
  std::mt19937_64 g {seed};
  std::normal_distribution<double> d;
  std::vector<T> v(r * c);
  for (auto& x : v) {
    if constexpr (std::is_same_v<T, double>) x = d(g); else x = T {d(g), d(g)};
  }
  matrix<T> m {r, c, std::move(v)};
  if (spd) {
    matrix<T> s {numerical::transpose(m) * m};
    std::vector<T> e(r * c);
    for (std::size_t i = 0; i < r; ++i) for (std::size_t j = 0; j < c; ++j) e[(i * c) + j] = s(i, j);
    for (std::size_t i = 0; i < r; ++i) e[(i * r) + i] += T(double(r));
    return matrix<T> {r, c, std::move(e)};
  }
  return m;
}

int main(int argc, char** argv)
{
  if (const char* t = std::getenv("NT")) numerical::set_max_threads(std::strtoul(t, nullptr, 10));
  const std::string op {argv[1]};
  const std::size_t n = std::strtoul(argv[2], nullptr, 10);
  const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
  auto a = rnd<double>(n, n, 1), b = rnd<double>(n, n, 2);
  auto za = rnd<std::complex<double>>(n, n, 3), zb = rnd<std::complex<double>>(n, n, 4);
  auto spd = op == "cholesky" ? rnd<double>(n, n, 5, true) : a;
  auto tall = rnd<double>(2 * n, n, 6);
  auto rhs = rnd<double>(n, 4, 7);
  double sink = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    if (op == "matmul") sink += (a * b)(0, 0);
    else if (op == "zmatmul") sink += (za * zb)(0, 0).real();
    else if (op == "solve") sink += numerical::solve(a, rhs)(0, 0);
    else if (op == "lu") sink += numerical::lu_decomposition<double>(a).determinant();
    else if (op == "inverse") sink += numerical::inverse(a)(0, 0);
    else if (op == "cholesky") sink += numerical::cholesky_decomposition<double>(spd).lower()(0, 0);
    else if (op == "qr") sink += numerical::qr_decomposition<double>(a).r()(0, 0);
    else if (op == "eigh") { auto s = spd; sink += numerical::symmetric_eigen(numerical::matrix<double>(a + numerical::transpose(a))).values()[0]; }
    else if (op == "svd") sink += numerical::svd_decomposition<double>(a).singular_values()[0];
    else if (op == "lstsq") sink += numerical::lstsq(tall, matrix<double>(rnd<double>(2 * n, 1, 8)))(0, 0);
  }
  auto t1 = std::chrono::steady_clock::now();
  std::printf("%s n=%zu : %.3f ms/appel (%g)\n", op.c_str(), n, std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, sink);
}
