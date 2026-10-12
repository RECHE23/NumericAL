// detail::parallel_for: every index runs once, sequentially when its work pays for fewer than two threads
// and across threads above, and an exception a worker throws comes back to the caller once every worker
// has joined -- one escaping a thread would call std::terminate.
#include <atomic>
#include <cstdlib>
#include <cstddef>
#include <mutex>
#include <set>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/parallel.hpp"

#if !defined(_WIN32)
#  include <sys/wait.h>
#  include <unistd.h>
#endif

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

// A parallel_for nested in a body finds the pool busy and runs its chunks in the calling worker: every index of
// both levels once, no wait on itself.
TEST(a_parallel_for_nested_in_a_body_runs_in_its_thread)
{
  std::vector<std::atomic<int>> hits(std::size_t {64} *64);
  numerical::detail::parallel_for(64, std::size_t {1} << 20, [&hits](std::size_t i) {
                                    numerical::detail::parallel_for(64, std::size_t {1} << 20, [&hits, i](std::size_t j) {
                                                                      ++hits[(i * 64) + j];
                                                                    });
                                  });
  bool once {true};
  for (const std::atomic<int>& hit : hits) {
    once = once && hit.load() == 1;
  }
  EXPECT(once);
}

// Two threads at once: one holds the pool, the other runs its chunks itself; both cover every index once.
TEST(parallel_fors_from_two_threads_both_complete)
{
  std::vector<std::atomic<int>> first(4096);
  std::vector<std::atomic<int>> second(4096);
  std::thread                   other {[&second] {
                                         for (int round {0}; round < 50; ++round) {
                                           numerical::detail::parallel_for(second.size(), std::size_t {1} << 16,
                                                                           [&second](std::size_t i) { ++second[i]; });
                                         }
                                       }};
  for (int round {0}; round < 50; ++round) {
    numerical::detail::parallel_for(first.size(), std::size_t {1} << 16, [&first](std::size_t i) { ++first[i]; });
  }
  other.join();
  bool fifty {true};
  for (std::size_t i {0}; i < first.size(); ++i) {
    fifty = fifty && first[i].load() == 50 && second[i].load() == 50;
  }
  EXPECT(fifty);
}

// A cap of one runs every index on the caller; the cap lifted, the hardware threads come back.
TEST(set_max_threads_caps_the_threads_a_kernel_uses)
{
  const std::size_t hardware {numerical::max_threads()};
  numerical::set_max_threads(1);
  EXPECT_EQ(numerical::max_threads(), 1U);
  const std::thread::id         caller    {std::this_thread::get_id()};
  std::atomic<bool>             elsewhere {false};
  numerical::detail::parallel_for(1024, std::size_t {1} << 20, [&elsewhere, caller](std::size_t) {
                                    if (std::this_thread::get_id() != caller) {
                                      elsewhere = true;
                                    }
                                  });
  EXPECT(!elsewhere.load());
  numerical::set_max_threads(0);
  EXPECT_EQ(numerical::max_threads(), hardware);
  EXPECT(hardware >= 1U);
}

namespace {
  // One body type through every way parallel_for runs: sequential, on the pool, capped to the caller, and
  // nested in itself (the pool busy). Coverage counts each body type's instantiation apart.
  struct nested_counter
  {
    std::vector<std::atomic<int>>* hits;
    std::size_t                    width;
    bool                           nest;

    void operator()(std::size_t i) const
    {
      if (nest) {
        numerical::detail::parallel_for(width, std::size_t {1} << 20, nested_counter {.hits = hits, .width = width, .nest = false});
      }
      ++(*hits)[nest ? 0 : i];
    }
  };
} // namespace

TEST(one_body_type_runs_sequentially_on_the_pool_capped_and_nested)
{
  std::vector<std::atomic<int>> hits(64);
  numerical::detail::parallel_for(64, 1, nested_counter {.hits                     = &hits, .width = 64, .nest = false}); // too little work
  numerical::detail::parallel_for(64, std::size_t {1} << 20, nested_counter {.hits = &hits, .width = 64, .nest = false}); // the pool
  numerical::set_max_threads(1);
  numerical::detail::parallel_for(64, std::size_t {1} << 20, nested_counter {.hits = &hits, .width = 64, .nest = false}); // capped
  numerical::set_max_threads(2);
  numerical::detail::parallel_for(64, std::size_t {1} << 20, nested_counter {.hits = &hits, .width = 64, .nest = false}); // a cap below the pool
  numerical::set_max_threads(0);
  numerical::detail::parallel_for(8, std::size_t {1} << 20, nested_counter {.hits = &hits, .width = 64, .nest = true});   // nested: busy
  bool right {hits[0].load() == 4 + 8 + 8};                                                                               // four flat runs, eight nested inner runs, eight outer bodies
  for (std::size_t i {1}; i < hits.size(); ++i) {
    right = right && hits[i].load() == 4 + 8;
  }
  EXPECT(right);
}

// A cap above one holds too: no more threads than it take part in a run, the caller's included (every worker
// used to join every run, whatever the cap).
TEST(a_cap_above_one_bounds_the_threads_of_a_run)
{
  std::mutex                guard;
  std::set<std::thread::id> seen;
  numerical::set_max_threads(2);
  numerical::detail::parallel_for(256, std::size_t {1} << 20, [&guard, &seen](std::size_t) {
                                    const std::scoped_lock held {guard};
                                    seen.insert(std::this_thread::get_id());
                                    std::this_thread::yield();
                                  });
  numerical::set_max_threads(0);
  EXPECT(!seen.empty() && seen.size() <= 2U);
}

#if !defined(_WIN32)
// A child forked while the pool runs has none of its workers, and their locks may be held: it runs its kernels
// alone, to the end. The child exits 0 when its run completed and its sum is right; an alarm ends a hung one.
TEST(a_forked_child_runs_its_kernels_alone)
{
  std::atomic<bool> stop {false};
  std::thread       busy {[&stop] {
                            while (!stop.load()) {
                              std::vector<std::atomic<int>> hits(512);
                              numerical::detail::parallel_for(hits.size(), std::size_t {1} << 20, [&hits](std::size_t i) { ++hits[i]; });
                            }
                          }};
  bool              every_child_completed {true};
  for (int fork_round {0}; fork_round < 40; ++fork_round) {
    const pid_t child {fork()};
    if (child == 0) {
      (void)alarm(10);
      // The parent's pool does not serve the child: it declines, and the run happens on the caller.
      const auto nothing {[](const void*, std::size_t, std::size_t) {}};
      if (numerical::detail::thread_pool::shared().try_run(nothing, nullptr, 1, 1, 1, 1)) {
        std::exit(2); // NOLINT(concurrency-mt-unsafe): a forked child has this one thread
      }
      std::vector<std::atomic<int>> hits(512);
      numerical::detail::parallel_for(hits.size(), std::size_t {1} << 20, [&hits](std::size_t i) { ++hits[i]; });
      int total {0};
      for (const std::atomic<int>& hit : hits) {
        total += hit.load();
      }
      // exit, not _exit, so that a coverage build writes the child's profile
      std::exit(total == 512 ? 0 : 1); // NOLINT(concurrency-mt-unsafe): a forked child has this one thread
    }
    int status {0};
    (void)waitpid(child, &status, 0);
    every_child_completed = every_child_completed && WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }
  stop = true;
  busy.join();
  EXPECT(every_child_completed);
}
#endif
