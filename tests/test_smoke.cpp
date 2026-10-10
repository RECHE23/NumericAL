// Smoke test: the umbrella header includes and a vector is usable end to end.
#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

TEST(umbrella_header_works)
{
  const numerical::vector<int> v {1, 2, 3};
  EXPECT_EQ(v.size(), 3U);
  EXPECT_EQ(numerical::dot(v, v), 14); // 1 + 4 + 9
}
