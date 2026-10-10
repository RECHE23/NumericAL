// detail::parallel_for: every index runs once, sequentially below the threshold and across threads above
// it, and an exception a worker throws comes back to the caller once every worker has joined -- one
// escaping a thread would call std::terminate.
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/parallel.hpp"

TEST(parallel_for_runs_every_index_once)
{
  for (const std::size_t threshold : {std::size_t {1}, std::size_t {1000}}) {
    std::vector<std::atomic<int>> hits(257);
    numerical::detail::parallel_for(hits.size(), threshold, [&hits](std::size_t i) { ++hits[i]; });
    bool once {true};
    for (const std::atomic<int>& hit : hits) {
      once = once && hit.load() == 1;
    }
    EXPECT(once);
  }
}

TEST(parallel_for_returns_a_worker_exception_after_the_join)
{
  std::atomic<std::size_t> ran    {0};
  bool                     caught {false};
  try {
    numerical::detail::parallel_for(1000, 1, [&ran](std::size_t i) {
                                      ++ran;
                                      if (i == 500) {
                                        throw std::runtime_error("index 500");
                                      }
                                    });
  }
  catch (const std::runtime_error& error) {
    caught = std::string_view {error.what()} == "index 500";
  }
  EXPECT(caught);
  EXPECT(ran.load() >= 1U); // the other chunks ran, and every worker joined before the rethrow
}
