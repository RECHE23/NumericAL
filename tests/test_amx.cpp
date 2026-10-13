// The GEMM paths' guards: limit_gemm_path never selects a path this processor cannot run; the AMX (amx.hpp) is
// used only on a known M-series family and after its self-test; the AVX paths only where cpuid reports them.
#include <array>
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
  for (const numerical::gemm_path path : {numerical::gemm_path::avx2, numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
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
