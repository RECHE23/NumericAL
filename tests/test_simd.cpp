// The row kernels the decompositions share: the scaled-row updates (simd.hpp) equal, to the bit, the scalar
// statements, over native vectors or not; the dot product sums every term; recorded row operations applied by
// runs equal them applied one by one.
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  // Products that are not exactly representable, of both signs: a reordered or fused operation shows in the
  // last bits.
  template <typename T>
  std::vector<T> values(std::size_t count,
                        double      offset)
  {
    std::vector<T> out(count);
    for (std::size_t i {0}; i < count; ++i) {
      out[i] = T((1.0 / (static_cast<double>(i) + offset)) - 0.21);
    }
    return out;
  }

  template <typename T>
  bool updates_match_the_scalar_statements(std::size_t count)
  {
    const std::vector<T> x            {values<T>(count, 3.0)};
    const T              f            {T(1.0 / 7.0)};
    std::vector<T>       down         {values<T>(count, 5.0)};
    std::vector<T>       up           {down};
    std::vector<T>       down_wanted  {down};
    std::vector<T>       up_wanted    {down};
    std::vector<T>       right        {down};
    std::vector<T>       right_wanted {down};
    for (std::size_t i {0}; i < count; ++i) {
      down_wanted[i]  -= f * x[i];
      up_wanted[i]    += f * x[i];
      right_wanted[i] -= x[i] * f;
    }
    numerical::detail::subtract_product<T>(down.data(), f, x.data(), count); // the templates: double's overloads
    numerical::detail::add_product<T>(up.data(), f, x.data(), count);        // choose their units (the test below)
    numerical::detail::subtract_product<T>(right.data(), x.data(), f, count);
    return down == down_wanted && up == up_wanted && right == right_wanted;
  }
} // namespace

TEST(scaled_row_updates_equal_the_scalar_statements_to_the_bit)
{
  for (const std::size_t count : {std::size_t {0}, std::size_t {1}, std::size_t {37}}) {
    EXPECT(updates_match_the_scalar_statements<double>(count));
    EXPECT(updates_match_the_scalar_statements<float>(count));
    EXPECT(updates_match_the_scalar_statements<std::complex<double>>(count));
  }
}

// double's row updates on the units the matrix product's kernel takes, on every path this processor runs: one fused
// multiply-add per element on AVX2 or AVX-512, the scalar statement elsewhere; and its dot product sums every term.
TEST(double_row_updates_follow_the_kernel_path)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue;
    }
    for (const std::size_t count : {std::size_t {0}, std::size_t {5}, std::size_t {37}, std::size_t {70}}) {
      const std::vector<double> x           {values<double>(count, 3.0)};
      const double              f           {1.0 / 7.0};
      std::vector<double>       down        {values<double>(count, 5.0)};
      std::vector<double>       up          {down};
      std::vector<double>       right       {down};
      std::vector<double>       down_wanted {down};
      std::vector<double>       up_wanted   {down};
      const bool                fused       {path == numerical::gemm_path::avx2 || path == numerical::gemm_path::avx512};
      for (std::size_t i {0}; i < count; ++i) {
        down_wanted[i] = fused ? std::fma(-f, x[i], down_wanted[i]) : down_wanted[i] - (f * x[i]);
        up_wanted[i]   = fused ? std::fma(f, x[i], up_wanted[i]) : up_wanted[i] + (f * x[i]);
      }
      numerical::detail::subtract_product(down.data(), f, x.data(), count);
      numerical::detail::add_product(up.data(), f, x.data(), count);
      numerical::detail::subtract_product(right.data(), x.data(), f, count);
      EXPECT(down == down_wanted);
      EXPECT(up == up_wanted);
      EXPECT(right == down_wanted);
      std::vector<double> whole(count);
      std::vector<double> other(count);
      double              want {0.0};
      for (std::size_t i {0}; i < count; ++i) {
        whole[i]  = static_cast<double>(i % 7) - 3.0;
        other[i]  = static_cast<double>((i % 5) + 1);
        want     += whole[i] * other[i];
      }
      EXPECT(numerical::detail::dot_product(whole.data(), other.data(), count) == want);
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// Several terms into one row, held in registers on x86-64: each element as the term-by-term calls compute it, on
// every path, across every remainder of the vector loops (32, 8 and 16, 4 elements at a time).
TEST(row_combinations_match_their_terms_one_by_one)
{
  for (const std::size_t count : {std::size_t {0}, std::size_t {3}, std::size_t {8}, std::size_t {13}, std::size_t {32}, std::size_t {45}, std::size_t {70}}) {
    for (const std::size_t terms : {std::size_t {0}, std::size_t {1}, std::size_t {5}}) {
      const std::size_t   ldx {count + 3};
      std::vector<double> x   {values<double>((terms * ldx) + 1, 3.0)};
      std::vector<double> f(terms);
      for (std::size_t t {0}; t < terms; ++t) {
        f[t] = 1.0 / static_cast<double>(t + 7);
      }
      for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::avx2, numerical::gemm_path::avx512}) {
        if (numerical::limit_gemm_path(path) == path) { // on every path this processor runs
          std::vector<double> held   {values<double>(count, 5.0)};
          std::vector<double> wanted {held};
          for (std::size_t t {0}; t < terms; ++t) {
            numerical::detail::subtract_product(wanted.data(), f[t], x.data() + (t * ldx), count);
          }
          numerical::detail::subtract_products(held.data(), f.data(), x.data(), ldx, terms, count);
          EXPECT(held == wanted);
        }
      }
      (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
      std::vector<float> narrow(count, 2.0F);
      std::vector<float> narrow_wanted {narrow};
      std::vector<float> fx(x.size());
      std::vector<float> ff(terms);
      for (std::size_t i {0}; i < x.size(); ++i) {
        fx[i] = static_cast<float>(x[i]);
      }
      for (std::size_t t {0}; t < terms; ++t) {
        ff[t] = static_cast<float>(f[t]);
      }
      for (std::size_t t {0}; t < terms; ++t) {
        numerical::detail::subtract_product(narrow_wanted.data(), ff[t], fx.data() + (t * ldx), count);
      }
      numerical::detail::subtract_products(narrow.data(), ff.data(), fx.data(), ldx, terms, count);
      EXPECT(narrow == narrow_wanted);
    }
  }
}

// Eight rows at once, each with its own factors on the same rows of x: each element as subtract_products gives it
// row by row, on every path, across every remainder of the column loops (8 and 4 at a time, then one).
TEST(rows_together_match_their_rows_one_by_one)
{
  constexpr std::size_t rows {numerical::detail::product_rows};
  for (const std::size_t count : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {4}, std::size_t {5}, std::size_t {8}, std::size_t {11}, std::size_t {16}}) {
    for (const std::size_t terms : {std::size_t {0}, std::size_t {1}, std::size_t {6}}) {
      const std::size_t   ldx {count + 3};
      const std::size_t   ldy {count + 2};
      const std::size_t   ldf {terms + 1};
      std::vector<double> x   {values<double>((terms * ldx) + 1, 3.0)};
      std::vector<double> f   {values<double>((rows * ldf) + 1, 0.25)};
      for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::avx2, numerical::gemm_path::avx512}) {
        if (numerical::limit_gemm_path(path) == path) {
          std::vector<double> held   {values<double>(rows * ldy, 5.0)};
          std::vector<double> wanted {held};
          for (std::size_t r {0}; r < rows; ++r) {
            numerical::detail::subtract_products(wanted.data() + (r * ldy), f.data() + (r * ldf), x.data(), ldx, terms, count);
          }
          numerical::detail::subtract_rows(held.data(), ldy, f.data(), ldf, x.data(), ldx, terms, count);
          EXPECT(held == wanted);
        }
      }
      (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
    }
  }
}

// Complex terms into one row: past the portable path, each part of each term one fused multiply-add (Re f · x, then
// Im f times the swapped parts); on it, the template's whole complex products. Every remainder of the vector loops.
TEST(complex_row_combinations_take_each_part_in_order)
{
  using c = std::complex<double>;
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::avx2, numerical::gemm_path::avx512}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue;
    }
    for (const std::size_t count : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {7}, std::size_t {16}, std::size_t {21}, std::size_t {37}}) {
      for (const std::size_t terms : {std::size_t {0}, std::size_t {1}, std::size_t {4}}) {
        const std::size_t ldx {count + 2};
        std::vector<c>    x((terms * ldx) + 1);
        std::vector<c>    f(terms);
        for (std::size_t k {0}; k < x.size(); ++k) {
          x[k] = c {std::sin(static_cast<double>(k)), std::cos(static_cast<double>(k) * 0.3)};
        }
        for (std::size_t t {0}; t < terms; ++t) {
          f[t] = c {1.0 / static_cast<double>(t + 3), -1.0 / static_cast<double>(t + 5)};
        }
        std::vector<c> held(count, c {0.5, -0.25});
        std::vector<c> wanted        {held};
        if (path == numerical::gemm_path::portable) {
          for (std::size_t t {0}; t < terms; ++t) {
            numerical::detail::subtract_product(wanted.data(), f[t], x.data() + (t * ldx), count);
          }
        }
        else {
          for (std::size_t i {0}; i < count; ++i) {
            double real {wanted[i].real()};
            double imag {wanted[i].imag()};
            for (std::size_t t {0}; t < terms; ++t) {
              const c value {x[(t * ldx) + i]};
              real = std::fma(-f[t].real(), value.real(), real);
              imag = std::fma(-f[t].real(), value.imag(), imag);
              real = std::fma(f[t].imag(), value.imag(), real);
              imag = std::fma(-f[t].imag(), value.real(), imag);
            }
            wanted[i] = c {real, imag};
          }
        }
        numerical::detail::subtract_products(held.data(), f.data(), x.data(), ldx, terms, count);
        EXPECT(held == wanted);
        if (terms == 1) { // a single term either way round is one term of the combination
          std::vector<c> left(count, c {0.5, -0.25});
          std::vector<c> right         {left};
          numerical::detail::subtract_product(left.data(), f[0], x.data(), count);
          numerical::detail::subtract_product(right.data(), x.data(), f[0], count);
          EXPECT(left == held && right == held);
        }
      }
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// A complex row of a Hermitian product: out[j] += conj(row[j]) · v[i] for j < i, each part two fused multiply-adds
// past the portable path (the template's complex products on it), and Σⱼ≤ᵢ row[j] v[j], exact here in any order
// (multiples of a quarter). Every remainder of the vector loops.
TEST(complex_hermitian_rows_take_each_part_in_order)
{
  using c = std::complex<double>;
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::avx2, numerical::gemm_path::avx512}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue;
    }
    for (const std::size_t i : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {4}, std::size_t {9}, std::size_t {17}}) {
      std::vector<c> row(i + 1);
      std::vector<c> v(i + 1);
      for (std::size_t j {0}; j <= i; ++j) {
        row[j] = c {static_cast<double>(j % 5) - 2.0, static_cast<double>(j % 3) / 4.0};
        v[j]   = c {static_cast<double>(j % 7) / 2.0, 1.0 - static_cast<double>(j % 4)};
      }
      std::vector<c> out(i + 1, c {0.25, -0.5});
      std::vector<c> wanted       {out};
      c              sum          {};
      for (std::size_t j {0}; j < i; ++j) {
        if (path == numerical::gemm_path::portable) {
          wanted[j] += std::conj(row[j]) * v[i];
        }
        else {
          wanted[j] = c {std::fma(v[i].imag(), row[j].imag(), std::fma(v[i].real(), row[j].real(), wanted[j].real())),
                         std::fma(v[i].imag(), row[j].real(), std::fma(-v[i].real(), row[j].imag(), wanted[j].imag()))};
        }
      }
      for (std::size_t j {0}; j <= i; ++j) {
        sum += row[j] * v[j];
      }
      const c got {numerical::detail::hermitian_row(nullptr, row.data(), v.data(), out.data(), i)};
      EXPECT(got == sum);
      EXPECT(out == wanted);
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// Whole numbers small enough that every partial sum is exact: any order of summation gives the exact dot
// product, so the vector form (four accumulators, then the lanes, then the remainder) must equal it.
TEST(dot_product_sums_every_term)
{
  for (const std::size_t count : {std::size_t {0}, std::size_t {3}, std::size_t {64}, std::size_t {77}}) {
    std::vector<double>               x(count);
    std::vector<double>               y(count);
    std::vector<std::complex<double>> zx(count);
    std::vector<std::complex<double>> zy(count);
    double                            want {0.0};
    for (std::size_t i {0}; i < count; ++i) {
      x[i]   = static_cast<double>(i % 7) - 3.0;
      y[i]   = static_cast<double>(i % 5) + 1.0;
      zx[i]  = {x[i], 1.0};
      zy[i]  = {y[i], 0.0};
      want  += x[i] * y[i];
    }
    EXPECT(numerical::detail::dot_product(x.data(), y.data(), count) == want);
    EXPECT(numerical::detail::dot_product(zx.data(), zy.data(), count).real() == want);
  }
}

namespace {

  // The pivot search the LU takes: the vectors' leading elements, then the scalar search on the rest.
  template <typename T>
  std::size_t first_largest(const std::vector<T>& x)
  {
    auto        best {std::abs(x[0])};
    std::size_t at   {0};
    for (std::size_t i {numerical::detail::largest_magnitude_vectors(x.data(), x.size(), best, at)}; i < x.size(); ++i) {
      if (std::abs(x[i]) > best) {
        best = std::abs(x[i]);
        at   = i;
      }
    }
    return at;
  }

  // The scalar search alone: the first index of the largest magnitude, a NaN never larger.
  template <typename T>
  std::size_t first_largest_scalar(const std::vector<T>& x)
  {
    auto        best {std::abs(x[0])};
    std::size_t at   {0};
    for (std::size_t i {1}; i < x.size(); ++i) {
      if (std::abs(x[i]) > best) {
        best = std::abs(x[i]);
        at   = i;
      }
    }
    return at;
  }

  // Values from a few magnitudes of both signs, zeros of both signs among them: a largest one recurs in several
  // lanes, at indices before and after one another's; a NaN at \p nan_at when it is inside.
  template <typename T>
  std::vector<T> ties(std::size_t count,
                      std::size_t seed,
                      std::size_t nan_at)
  {
    constexpr std::array<double, 7> pool {-3.0, 2.0, -0.0, 3.0, 0.0, -2.0, 1.0};
    std::vector<T>                  out(count);
    for (std::size_t i {0}; i < count; ++i) {
      out[i] = T(pool[((i * 5) + (seed * 3) + ((i * i) % 11)) % pool.size()]);
    }
    if (nan_at < count) {
      out[nan_at] = T(std::numeric_limits<double>::quiet_NaN());
    }
    return out;
  }

  template <typename T>
  bool search_matches_the_scalar_one()
  {
    for (std::size_t count {1}; count <= 41; ++count) {
      for (std::size_t seed {0}; seed < 7; ++seed) {
        for (const std::size_t nan_at : {count, std::size_t {0}, count / 2, count - 1}) {
          const std::vector<T> x {ties<T>(count, seed, nan_at)};
          if (first_largest(x) != first_largest_scalar(x)) {
            return false;
          }
        }
      }
    }
    std::vector<T> later(19, T(1.0));
    later[13] = T(-4.0);
    later[17] = T(4.0);
    return first_largest(later) == 13;
  }
} // namespace

// The pivot search over native vectors finds the scalar search's index: the first of the largest magnitudes, across
// lanes that each hold one, a NaN never chosen (nor a NaN in x[0] ever left).
TEST(the_largest_magnitude_is_the_scalar_searchs_first)
{
  EXPECT(search_matches_the_scalar_one<double>());
  EXPECT(search_matches_the_scalar_one<float>());
  EXPECT(search_matches_the_scalar_one<std::complex<double>>());
}

// A symmetric product's row: the sum of every term and the scaled row added to out, in one pass, on every
// kernel this processor runs. Small integers: every order sums them exactly.
TEST(symmetric_row_sums_every_term_and_adds_the_scaled_row)
{
  using kernel = double (*)(const double*, const double*, double, double*, std::size_t);
  std::vector<kernel> kernels {[](const double* row, const double* v, double f, double* out, std::size_t count) {
                                 return numerical::detail::symmetric_row(row, v, f, out, count);
                               }};
#ifdef NUMERICAL_X86_COMPILED
  if (numerical::detail::cpu_features().avx2_fma) {
    kernels.push_back(&numerical::detail::avx2_symmetric_row);
  }
  if (numerical::detail::cpu_features().avx512f) {
    kernels.push_back(&numerical::detail::avx512_symmetric_row);
  }
#endif
  for (const kernel run : kernels) {
    for (const std::size_t count : {std::size_t {0}, std::size_t {3}, std::size_t {64}, std::size_t {77}, std::size_t {130}}) {
      std::vector<double> row(count);
      std::vector<double> v(count);
      std::vector<double> out(count);
      double              want {0.0};
      for (std::size_t i {0}; i < count; ++i) {
        row[i]  = static_cast<double>(i % 7) - 3.0;
        v[i]    = static_cast<double>(i % 5) + 1.0;
        out[i]  = static_cast<double>(i % 3);
        want   += row[i] * v[i];
      }
      EXPECT(run(row.data(), v.data(), 2.0, out.data(), count) == want);
      bool added {true};
      for (std::size_t i {0}; i < count; ++i) {
        added = added && out[i] == static_cast<double>(i % 3) + (2.0 * row[i]);
      }
      EXPECT(added);
    }
  }
  std::vector<std::complex<double>> z {{1.0, 2.0}, {3.0, -1.0}};
  std::vector<std::complex<double>> zout(2);
  EXPECT(numerical::detail::symmetric_row(z.data(), z.data(), std::complex<double> {2.0, 0.0}, zout.data(), 2)
         == (z[0] * z[0]) + (z[1] * z[1]));
  EXPECT(zout[1] == z[1] * 2.0);
}

namespace {

  // symmetric_rows checked against its definition, on rows of small integers every order sums exactly: each row's
  // sum to its diagonal, and its earlier columns of out scaled by its own element of v.
  template <typename T,
            typename Kernel>
  bool symmetric_rows_match(Kernel run)
  {
    bool matches {true};
    for (const std::size_t i : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {8}, std::size_t {13},
                                std::size_t {64}, std::size_t {77}}) {
      const std::size_t n {i + numerical::detail::symmetric_row_group + 2};
      std::vector<T>    rows(numerical::detail::symmetric_row_group * n);
      std::vector<T>    v(n);
      std::vector<T>    out(n);
      for (std::size_t k {0}; k < rows.size(); ++k) {
        rows[k] = static_cast<T>(static_cast<int>(k % 7) - 3);
      }
      for (std::size_t k {0}; k < n; ++k) {
        v[k]   = static_cast<T>((k % 5) + 1);
        out[k] = static_cast<T>(k % 3);
      }
      std::vector<T>                                          want_out {out};
      std::array<T, numerical::detail::symmetric_row_group>   want     {};
      for (std::size_t r {0}; r < numerical::detail::symmetric_row_group; ++r) {
        for (std::size_t k {0}; k <= i + r; ++k) {
          want[r] += rows[(r * n) + k] * v[k];
          if (k < i + r) {
            want_out[k] += rows[(r * n) + k] * v[i + r];
          }
        }
      }
      matches = matches && run(rows.data(), n, v.data(), out.data(), i) == want && out == want_out;
    }
    return matches;
  }
} // namespace

// Four rows of a symmetric product at once, on every kernel this processor runs, and for types with native vectors
// and without.
TEST(symmetric_rows_sum_each_row_and_add_each_scaled_row)
{
  EXPECT(symmetric_rows_match<double>([](const double* rows, std::size_t n, const double* v, double* out, std::size_t i) {
                                        return numerical::detail::symmetric_rows(rows, n, v, out, i);
                                      }));
  EXPECT(symmetric_rows_match<float>([](const float* rows, std::size_t n, const float* v, float* out, std::size_t i) {
                                       return numerical::detail::symmetric_rows(rows, n, v, out, i);
                                     }));
  EXPECT(symmetric_rows_match<long double>(
           [](const long double* rows, std::size_t n, const long double* v, long double* out, std::size_t i) {
             return numerical::detail::symmetric_rows(rows, n, v, out, i);
           }));
#ifdef NUMERICAL_X86_COMPILED
  if (numerical::detail::cpu_features().avx2_fma) {
    EXPECT(symmetric_rows_match<double>(&numerical::detail::avx2_symmetric_rows));
  }
  if (numerical::detail::cpu_features().avx512f) {
    EXPECT(symmetric_rows_match<double>(&numerical::detail::avx512_symmetric_rows));
  }
#endif
}

namespace {

  using numerical::detail::row_operation;

  row_operation rotation(std::size_t i,
                         std::size_t j,
                         double      angle)
  {
    return {.what = row_operation::kind::rotate, .i = i, .j = j, .c = std::cos(angle), .s = std::sin(angle)};
  }

  // The operations one at a time, each on whole rows: the definition the runs must reproduce to the bit.
  std::vector<double> applied_one_by_one(std::vector<double>               m,
                                         std::size_t                       n,
                                         const std::vector<row_operation>& operations)
  {
    for (const row_operation& op : operations) {
      for (std::size_t k {0}; k < n; ++k) {
        double& a {m[(op.i * n) + k]};
        double& b {m[(op.j * n) + k]};
        if (op.what == row_operation::kind::rotate) {
          const double x {a};
          const double y {b};
          a = (op.c * x) + (op.s * y);
          b = (op.c * y) - (op.s * x);
        }
        else if (op.what == row_operation::kind::negate) {
          a = -a;
        }
        else {
          std::swap(a, b);
        }
      }
    }
    return m;
  }
} // namespace

// Chained rotations going down (as a QL sweep records them) or up (as a QR sweep), a run ending the list, and
// operations that chain to nothing: split into runs, and applied by runs exactly as one by one.
TEST(rotation_runs_apply_as_the_operations_one_by_one)
{
  const std::vector<row_operation> down {rotation(3, 4, 0.3), rotation(2, 3, -1.1), rotation(1, 2, 0.7)};
  const std::vector<row_operation> up   {rotation(0, 1, 0.2), rotation(1, 2, 2.1), rotation(2, 3, -0.4),
                                         {.what = row_operation::kind::negate, .i = 2, .j = 2, .c = 0.0, .s = 0.0},
                                         {.what = row_operation::kind::swap, .i = 0, .j = 4, .c = 0.0, .s = 0.0},
                                         rotation(0, 3, 0.9), rotation(1, 2, 0.5)};
  const auto down_runs {numerical::detail::rotation_runs(down)};
  const auto up_runs   {numerical::detail::rotation_runs(up)};
  EXPECT(down_runs.size() == 1U && down_runs[0].count == 3 && down_runs[0].down);
  EXPECT(up_runs.size() == 5U && up_runs[0].count == 3 && !up_runs[0].down);
  for (const std::size_t n : {std::size_t {5}, std::size_t {70}}) { // one chunk of columns, then two
    std::vector<double> m(n * n);
    for (std::size_t i {0}; i < m.size(); ++i) {
      m[i] = (1.0 / static_cast<double>(i + 3)) - 0.17;
    }
    for (const std::vector<row_operation>* operations : {&down, &up}) {
      numerical::buffer<double> runs(m.begin(), m.end());
      numerical::detail::apply_row_operations(runs, n, *operations);
      EXPECT(runs == applied_one_by_one(m, n, *operations));
    }
  }
}

// operator_times: the complex product std::operator* computes, finite and not (an infinity by zero, whose naive parts are both
// NaN, takes the operator's own rescue), for both precisions; a real product as the operator.
TEST(operator_times_is_the_operator_s_product)
{
  const auto same = []<typename T>(const T& a, const T& b) {
                      const T    got {numerical::detail::operator_times(a, b)};
                      const T    want {a * b};
                      const auto equal = [](auto x, auto y) { return x == y || (std::isnan(x) && std::isnan(y)); };
                      return equal(got.real(), want.real()) && equal(got.imag(), want.imag());
                    };
  const double inf {std::numeric_limits<double>::infinity()};
  const double nan {std::numeric_limits<double>::quiet_NaN()};
  using zd = std::complex<double>;
  using cf = std::complex<float>;
  EXPECT(same(zd {0.1, -0.3}, zd {1.7, 2.9}));
  EXPECT(same(zd {inf, 0.0}, zd {0.0, 0.0}));      // ∞·0 in both naive parts: the rescue
  EXPECT(same(zd {inf, inf}, zd {nan, 1.0}));
  EXPECT(same(zd {inf, 1.0}, zd {1.0, inf}));      // ∞ − ∞ in the real part alone: no rescue
  EXPECT(same(zd {1e300, 1e300}, zd {1e300, -1e300}));
  EXPECT(same(cf {0.1F, -0.3F}, cf {1.7F, 2.9F}));
  EXPECT(same(cf {static_cast<float>(inf), 0.0F}, cf {0.0F, 0.0F}));
  EXPECT(numerical::detail::operator_times(0.1, 3.0) == 0.1 * 3.0);
}
