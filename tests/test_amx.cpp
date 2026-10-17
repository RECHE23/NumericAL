// The GEMM paths' guards: limit_gemm_path never selects a path this processor cannot run; the AMX (amx.hpp) is
// used only on a known M-series family and after its self-test; the AVX paths only where cpuid reports them.
#include <array>
#include <cmath>
#include <vector>
#include <cstdint>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

TEST(limit_gemm_path_takes_the_most_capable_path_that_runs_here)
{
  // The portable path always runs; every other one where the processor has it; a limit above what runs falls to
  // the most capable below it, and the highest limit restores the best.
  EXPECT(numerical::limit_gemm_path(numerical::gemm_path::portable) == numerical::gemm_path::portable);
  EXPECT(numerical::gemm_path_in_use() == numerical::gemm_path::portable);
  EXPECT(numerical::detail::gemm_path_usable(numerical::gemm_path::portable));
  for (const numerical::gemm_path path : {numerical::gemm_path::neon, numerical::gemm_path::avx2, numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    const bool                 usable {numerical::detail::gemm_path_usable(path)};
    const numerical::gemm_path got    {numerical::limit_gemm_path(path)};
    EXPECT(usable ? got == path : got < path);
  }
  EXPECT(numerical::gemm_path_in_use() == numerical::detail::best_gemm_path(numerical::gemm_path::amx));
#ifndef NUMERICAL_AMX_COMPILED
  EXPECT(!numerical::detail::amx_usable()); // compiled out: never in use
#endif
}

#ifdef NUMERICAL_AMX_COMPILED
TEST(only_the_m1_to_m4_families_are_known)
{
  for (const std::uint32_t family : {0x1b588bb3U, 0xda33d83dU, 0xfa33415eU, 0x5f4dea93U, 0x72015832U, 0x6f5129acU, 0x17d5b93aU}) {
    EXPECT(numerical::detail::amx_family_known(family));
  }
  // An iPhone family, an older Mac family and no answer at all are not.
  for (const std::uint32_t family : {0x8765edeaU, 0x462504d2U, 0U}) {
    EXPECT(!numerical::detail::amx_family_known(family));
    EXPECT(!numerical::detail::amx_usable_on(family, false)); // and its self-test, which could fault, never runs
  }
}

TEST(a_virtual_machine_does_not_use_the_amx)
{
  // Its host decides what the guest's processor offers: even a known family is not used there.
  EXPECT(!numerical::detail::amx_usable_on(0x1b588bb3U, true));
}

TEST(this_chip_is_known_and_its_amx_passes_the_self_test)
{
  // The local gate runs on an M-series Mac: the path it measures is the one users get.
  EXPECT(numerical::detail::amx_family_known(numerical::detail::cpu_family()));
  EXPECT(numerical::detail::amx_self_test());
  EXPECT(numerical::detail::amx_usable());
}

namespace {
  // y = A x by the AMX's lanes: element j of a row on lane j mod 8, each lane a chain of fused multiply-adds in
  // ascending order, the lanes added from 0 to 7.
  std::vector<double> lane_product(const std::vector<double>& a,
                                   std::size_t                rows,
                                   std::size_t                cols,
                                   const std::vector<double>& x)
  {
    std::vector<double> y(rows);
    for (std::size_t r {0}; r < rows; ++r) {
      std::array<double, 8> lane {};
      for (std::size_t j {0}; j < cols; ++j) {
        lane[j % 8] = std::fma(a[(r * cols) + j], x[j], lane[j % 8]);
      }
      double sum {lane[0]};
      for (std::size_t l {1}; l < 8; ++l) {
        sum += lane[l];
      }
      y[r] = sum;
    }
    return y;
  }

  // Whether matrix × vector of the shape, on the path in use, equals the AMX's lanes to the bit (\p lanes) or, off
  // them, the plain sum within a rounding.
  bool matvec_as_expected(std::size_t rows,
                          std::size_t cols,
                          bool        lanes)
  {
    std::vector<double> a(rows * cols);
    std::vector<double> x(cols);
    for (std::size_t i {0}; i < a.size(); ++i) {
      a[i] = std::sin(static_cast<double>(i));
    }
    for (std::size_t j {0}; j < cols; ++j) {
      x[j] = std::cos(static_cast<double>(j));
    }
    const numerical::vector<double> got  {numerical::matrix<double> {rows, cols, a} *numerical::vector<double> {std::vector<double>(x)}};
    const std::vector<double>       want {lane_product(a, rows, cols, x)};
    bool                            same {true};
    for (std::size_t r {0}; r < rows; ++r) {
      same = same && (lanes ? got[r] == want[r] : std::abs(got[r] - want[r]) <= 1e-9 * static_cast<double>(cols));
    }
    return same;
  }
} // namespace

// On the AMX path a double matrix of 2^15 to 2^18 elements is multiplied by a vector on the AMX, its lanes to the
// bit, with rows and columns past a multiple of eight. Smaller, larger, rows longer than a dot chunk, or off the AMX
// path, the NEON rows answer, within a rounding.
TEST(matrix_times_vector_runs_on_the_amx_lanes_where_it_pays)
{
  EXPECT(matvec_as_expected(203, 181, true));    // 36 743 elements, one block, tails of rows and columns
  EXPECT(matvec_as_expected(511, 509, true));    // 260 099, just under 2^18
  EXPECT(matvec_as_expected(100, 100, false));   // under 2^15
  EXPECT(matvec_as_expected(512, 512, false));   // 2^18
  EXPECT(matvec_as_expected(3, 20000, false));   // rows longer than a dot chunk
  (void)numerical::limit_gemm_path(numerical::gemm_path::neon);
  EXPECT(matvec_as_expected(203, 181, false));
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}
#endif

#ifdef NUMERICAL_X86_COMPILED
TEST(the_avx_paths_need_the_instructions_and_the_saved_registers)
{
  using numerical::detail::decode_features;
  constexpr std::uint32_t fma     {(1U << 12U) | (1U << 28U)}; // FMA, with the AVX bit
  constexpr std::uint32_t osxsave {1U << 27U};
  constexpr std::uint32_t avx2    {1U << 5U};
  constexpr std::uint32_t avx512f {1U << 16U};
  EXPECT(decode_features(fma | osxsave, avx2 | avx512f, 0xe6U).avx2_fma);
  EXPECT(decode_features(fma | osxsave, avx2 | avx512f, 0xe6U).avx512f);
  EXPECT(!decode_features(fma | osxsave, avx2 | avx512f, 0x6U).avx512f);           // the system saves no 512-bit state
  EXPECT(!decode_features(fma | osxsave, avx2, 0x2U).avx2_fma);                    // nor the 256-bit one
  EXPECT(!decode_features(osxsave | (1U << 28U), avx2, 0x6U).avx2_fma);            // no FMA
  EXPECT(!decode_features((1U << 12U) | osxsave, avx2 | avx512f, 0xe6U).avx2_fma); // no AVX instructions
  EXPECT(!decode_features((1U << 12U) | osxsave, avx2 | avx512f, 0xe6U).avx512f);
  EXPECT(!decode_features(fma | osxsave, 0U, 0xe6U).avx2_fma);                     // no AVX2
  EXPECT(numerical::detail::saved_state(0U) == 0U);                                // no OSXSAVE: no xgetbv
}
#endif
