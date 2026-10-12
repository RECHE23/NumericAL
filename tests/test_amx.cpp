// The AMX path's guards (amx.hpp): it is used only on a known M-series family and after its self-test, and
// set_amx turns it on only where it can run; on any other target it is absent and set_amx reports so.
#include <cstdint>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

TEST(set_amx_turns_the_path_on_only_where_it_can_run)
{
  const bool before {numerical::amx_enabled()};
  EXPECT(!numerical::set_amx(false));
  EXPECT(!numerical::amx_enabled());
  const bool usable {numerical::detail::amx_usable()};
  EXPECT(numerical::set_amx(true) == usable);
  EXPECT(numerical::amx_enabled() == usable);
  (void)numerical::set_amx(before);
#ifndef NUMERICAL_AMX_COMPILED
  EXPECT(!usable); // compiled out: never in use
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
    EXPECT(!numerical::detail::amx_usable_on(family)); // and its self-test, which could fault, never runs
  }
}

TEST(this_chip_is_known_and_its_amx_passes_the_self_test)
{
  // The local gate runs on an M-series Mac: the path it measures is the one users get.
  EXPECT(numerical::detail::amx_family_known(numerical::detail::cpu_family()));
  EXPECT(numerical::detail::amx_self_test());
  EXPECT(numerical::detail::amx_usable());
}
#endif
