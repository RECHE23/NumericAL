// NumericAL GEMM micro-benchmark — the shipping matrix product (numerical::operator*)
// against two bench-local references, to document its cache-friendly loop order
// and guard against a regression. Build and run with `make bench-gemm`.
//
//   * naive ijk : the textbook order — for each (i,j) sum over k. b(k,j) is read
//                 down a column (stride = b.cols()), so every step is a fresh
//                 cache line. This is what operator* used to do; it is the
//                 baseline the speed-up column is measured against.
//   * operator* : the shipping product (matrix.hpp → gemm.hpp): packed panels and
//                 a register tile, every element summed in ascending order.
//   * ikj       : the i-k-j loop operator* ran before the packed kernel
//                 (detail::gemm_rows): the same sums in the same order, without
//                 packing or a register tile — the speed-up the kernel buys.
//
// All three share the same numerical::detail::parallel_for, so the comparison
// isolates the inner loop order. The shipping operator* is also compared against
// the einsum tile_gemm path ("ij,jk->ik") to settle the two-matmul-paths question.
// Results are checked equal before timing.
//
// This program only collects raw per-call samples and emits the canonical sciforge.bench
// JSON (emit_case/emit_run); benchmarks/gemm_bench.py formats the table. The timing and the
// JSON emitter live in SciForge's shared C++ collector (include/sciforge/bench.hpp).
#include <numerical/tensor.hpp> // brings matrix, tensor and einsum

#include <sciforge/bench.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using numerical::einsum;
using numerical::matrix;
using numerical::tensor;

namespace {

matrix<double> random_matrix(std::size_t n, std::mt19937_64& rng)
{
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double>                    v(n * n);
  for (double& x : v) { x = dist(rng); }
  return matrix<double> {n, n, std::move(v)};
}

// The baseline: the textbook i-j-k order with b accessed down a column. This is
// the loop nest operator* used before the i-k-j reorder.
std::vector<double> gemm_ijk(const matrix<double>& a, const matrix<double>& b)
{
  const std::size_t   n {a.rows()}, m {b.cols()}, p {a.cols()};
  std::vector<double> out(n * m);
  numerical::detail::parallel_for(n, 64, [&](std::size_t i) {
                                 for (std::size_t j {0}; j < m; ++j) {
                                   double sum {0.0};
                                   for (std::size_t k {0}; k < p; ++k) { sum += a(i, k) * b(k, j); }
                                   out[(i * m) + j] = sum;
                                 }
                               });
  return out;
}

// The i-k-j loop operator* ran before the packed kernel, parallel over the rows of C: the same sum, in
// the same order, without packing or a register tile.
std::vector<double> gemm_ikj(const matrix<double>& a, const matrix<double>& b)
{
  std::vector<double> out(a.rows() * b.cols(), 0.0);
  numerical::detail::gemm_rows(a.rows(), b.cols(), a.cols(), &*a.begin(), &*b.begin(), out.data());
  return out;
}

double sum_matrix(const matrix<double>& c) { double s {0}; for (double x : c) { s += x; } return s; }
double sum_vec(const std::vector<double>& v) { double s {0}; for (double x : v) { s += x; } return s; }

bool matches(const matrix<double>& ref, const std::vector<double>& v)
{
  if (ref.rows() * ref.cols() != v.size()) { return false; }
  double      maxerr {0};
  std::size_t i {0};
  for (double x : ref) { maxerr = std::max(maxerr, std::abs(x - v[i++])); }
  return maxerr < 1e-6;
}
} // namespace

int main()
{
  std::mt19937_64          rng {12345};
  std::vector<std::string> cases;

  for (const std::size_t N : {128u, 256u, 512u, 1024u, 2048u}) {
    // Keep the original per-N repetition count as the sample count: the box-plots/CI gain of
    // many samples is a runtime trade-off to measure first (deferred), not a parity change —
    // we do NOT collect 40 samples of a 2048³ GEMM.
    const int            reps {N <= 256 ? 5 : (N <= 1024 ? 3 : 1)};
    const matrix<double> a {random_matrix(N, rng)};
    const matrix<double> b {random_matrix(N, rng)};
    const tensor<double> ta {a};
    const tensor<double> tb {b};
    const matrix<double> shipped {a * b}; // the product under test = the correctness reference

    // One case per variant: collect raw seconds (inner=1 ⇒ 1 sample = 1 GEMM, no auto-calibrate
    // batching), check the result against the shipped product, emit a canonical Case. The thunk's
    // double result is observed inside collect (do_not_optimize) — no magic sink needed.
    const auto add_case = [&](const std::string& variant,
                              auto&&             thunk,
                              bool               ok) {
                            const std::vector<double> samples =
                              sciforge::bench::collect(thunk, reps, 1);
                            const std::string name = "N=" + std::to_string(N) + " " + variant;
                            cases.push_back(sciforge::bench::emit_case(
                                              name, "s", samples,
                                              {{"n", sciforge::bench::json_number(static_cast<double>(N))},
                                               {"variant", sciforge::bench::json_string(variant)}},
                                              ok ? "true" : "false")); // result = a raw JSON boolean
                          };

    add_case("naive-ijk", [&] { return sum_vec(gemm_ijk(a, b)); },
             matches(shipped, gemm_ijk(a, b)));
    add_case("operator*", [&] { return sum_matrix(a * b); },
             true); // operator* IS the reference product
    add_case("einsum", [&] { return sum_vec(einsum("ij,jk->ik", ta, tb).data()); },
             matches(shipped, einsum("ij,jk->ik", ta, tb).data()));
    add_case("ikj", [&] { return sum_vec(gemm_ikj(a, b)); },
             matches(shipped, gemm_ikj(a, b)));
  }

  const std::string meta = sciforge::bench::json_object(
    {{"bench", sciforge::bench::json_string("gemm")},
     {"compiler", sciforge::bench::json_string(__VERSION__)}});
  std::printf("%s\n", sciforge::bench::emit_run(meta, cases).c_str());
  return 0;
}
