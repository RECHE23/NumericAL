// detail::parallel_for: every index runs once, sequentially when its work pays for fewer than two threads
// and across threads above, and an exception a worker throws comes back to the caller once every worker
// has joined -- one escaping a thread would call std::terminate.
#include <atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/parallel.hpp"

TEST(parallel_for_runs_every_index_once)
{
  for (const std::size_t work : {std::size_t {1}, std::size_t {1} << 20}) { // sequential, then threaded
    std::vector<std::atomic<int>> hits(257);
    numerical::detail::parallel_for(hits.size(), work, [&hits](std::size_t i) { ++hits[i]; });
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
    numerical::detail::parallel_for(1000, std::size_t {1} << 20, [&ran](std::size_t i) {
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

// A thread is started for each share of work that pays for it (detail::work_per_thread), at most one per
// index and per hardware thread: an inverse of order 16 (16 solves of 256) starts none, where it started
// one per hardware thread and took 177 µs instead of 8.
TEST(threads_for_pays_a_thread_for_each_share_of_work)
{
  using numerical::detail::threads_for;
  constexpr std::size_t most {std::numeric_limits<std::size_t>::max()};
  EXPECT_EQ(threads_for(16, 256, 8), 0U);                                      // 4096 multiply-adds: none
  EXPECT_EQ(threads_for(16, 16384, 8), 4U);                                    // 2^18: four shares
  EXPECT_EQ(threads_for(256, 65536, 8), 8U);                                   // as many as the hardware has
  EXPECT_EQ(threads_for(3, std::size_t {1} << 20, 8), 3U);                     // one per index at most
  EXPECT_EQ(threads_for(most, most, 8), 8U);                                   // a product past size_t saturates
  EXPECT_EQ(threads_for(std::size_t {1} << 20, std::size_t {1} << 20, 0), 1U); // an unknown hardware: one
  EXPECT_EQ(threads_for(5, 0, 8), 0U);                                         // no work, no thread
}
