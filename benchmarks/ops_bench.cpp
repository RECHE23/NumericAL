// NumericAL decomposition micro-benchmark for `make bench-diff`: one operation over several orders, each the
// best of five timed batches, on deterministic inputs. It is compiled against two include trees (a release and
// the working tree) and run alternately by benchmarks/bench_diff.py, so it uses only what every release has.
//
//   ops_bench <threads> <op> <order>...    prints "order:microseconds" per order, on one line
//
// op is one of mm, solve, inv, lu, lsolve, chol, qr, eigh, svd. lsolve solves four right-hand sides with a
// factor built once; solve factors and solves.
#include <numerical/numerical.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>
#include <vector>

namespace {

  //! \brief Keeps \p value alive past the optimiser.
  template <typename T>
  void keep(const T& value)
  {
    asm volatile ("" : : "r"(&value) : "memory");
  }

  //! \brief The best of five batches of \p run, in microseconds per call; a batch holds about 1e8 / n³ calls.
  template <typename Run>
  double best_of(const Run& run, std::size_t n)
  {
    const std::size_t calls {(100000000 / (n * n * n)) + 1};
    double            best {1e300};
    for (int batch {0}; batch < 5; ++batch) {
      const auto start {std::chrono::steady_clock::now()};
      for (std::size_t call {0}; call < calls; ++call) {
        run();
      }
      const std::chrono::duration<double, std::micro> spent {std::chrono::steady_clock::now() - start};
      best = std::min(best, spent.count() / static_cast<double>(calls));
    }
    return best;
  }
} // namespace

int main(int argc, char** argv)
{
  if (argc < 4) {
    std::fprintf(stderr, "usage: ops_bench <threads> <op> <order>...\n");
    return 2;
  }
  numerical::set_max_threads(static_cast<std::size_t>(std::atoi(argv[1])));
  const std::string_view op {argv[2]};
  for (int arg {3}; arg < argc; ++arg) {
    const auto              n {static_cast<std::size_t>(std::atoi(argv[arg]))};
    std::mt19937_64         generator {1};
    std::normal_distribution<double> normal;
    std::vector<double>     values(n * n);
    std::vector<double>     rhs(n * 4);
    std::vector<double>     spd(n * n);
    for (double& x : values) {
      x = normal(generator);
    }
    for (double& x : rhs) {
      x = normal(generator);
    }
    for (std::size_t i {0}; i < n; ++i) { // A Aᵀ + n I: symmetric positive definite
      for (std::size_t j {0}; j < n; ++j) {
        double sum {i == j ? static_cast<double>(n) : 0.0};
        for (std::size_t k {0}; k < n; ++k) {
          sum += values[(i * n) + k] * values[(j * n) + k];
        }
        spd[(i * n) + j] = sum;
      }
    }
    const numerical::matrix<double> m(n, n, values);
    const numerical::matrix<double> b(n, 4, rhs);
    const numerical::matrix<double> s(n, n, spd);
    double                          time {-1.0};
    if (op == "mm") {
      time = best_of([&] { keep(m * m); }, n);
    }
    else if (op == "solve") {
      time = best_of([&] { keep(numerical::lu_decomposition<double>(m).solve(b)); }, n);
    }
    else if (op == "inv") {
      time = best_of([&] { keep(numerical::inverse(m)); }, n);
    }
    else if (op == "lu") {
      time = best_of([&] { keep(numerical::lu_decomposition<double>(m)); }, n);
    }
    else if (op == "lsolve") {
      const numerical::lu_decomposition<double> factor(m);
      time = best_of([&] { keep(factor.solve(b)); }, n);
    }
    else if (op == "chol") {
      time = best_of([&] { keep(numerical::cholesky_decomposition<double>(s)); }, n);
    }
    else if (op == "qr") {
      time = best_of([&] { keep(numerical::qr_decomposition<double>(m)); }, n);
    }
    else if (op == "eigh") {
      time = best_of([&] { keep(numerical::symmetric_eigen(s)); }, n);
    }
    else if (op == "svd") {
      time = best_of([&] { keep(numerical::svd_decomposition<double>(m)); }, n);
    }
    else {
      std::fprintf(stderr, "ops_bench: unknown op '%s'\n", argv[2]);
      return 2;
    }
    std::printf(" %zu:%.4f", n, time);
  }
  std::printf("\n");
  return 0;
}
