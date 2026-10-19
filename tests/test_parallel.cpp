// detail::parallel_for: every index runs once, sequentially when its work pays for fewer than two threads
// and across threads above, and an exception a worker throws comes back to the caller once every worker
// has joined -- one escaping a thread would call std::terminate.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <mutex>
#include <set>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/parallel.hpp"

#if !defined(_WIN32)
#  include <sys/wait.h>
#  include <unistd.h>

#  if defined(__SANITIZE_ADDRESS__)
#    define NUMERICAL_TEST_ADDRESS_SANITIZER 1
#  elif defined(__has_feature)
#    if __has_feature(address_sanitizer)
#      define NUMERICAL_TEST_ADDRESS_SANITIZER 1
#    endif
#  endif

namespace {

  // Ends a forked child with its verdict. The parent's other threads do not exist in the child, so what their stacks
  // held looks leaked to LeakSanitizer, whose check std::exit runs: under the address sanitizer the child leaves by
  // _exit. Elsewhere std::exit, so that a coverage build writes the child's profile.
  [[noreturn]] void end_child(int code)
  {
#  if defined(NUMERICAL_TEST_ADDRESS_SANITIZER)
    _exit(code);
#  else
    std::exit(code); // NOLINT(concurrency-mt-unsafe): a forked child has this one thread
#  endif
  }
} // namespace
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

namespace {
  // Every index's hits, and the task's, of runs beside a loop.
  struct beside_counts
  {
    std::vector<std::atomic<int>> hits;
    std::atomic<int>              side {0};
  };

  // One body type and one task type for every way a run goes (threaded, capped, sequential, nested in a busy pool),
  // so that one instantiation meets them all.
  void run_beside(beside_counts& counts,
                  std::size_t    work)
  {
    numerical::detail::parallel_for_beside(counts.hits.size(), work, [&counts](std::size_t i) { ++counts.hits[i]; },
                                           [&counts] { ++counts.side; });
  }

  // Whether every index ran \p times and the task \p sides times.
  bool every_index(const beside_counts& counts,
                   int                  times,
                   int                  sides)
  {
    bool all {counts.side.load() == sides};
    for (const std::atomic<int>& hit : counts.hits) {
      all = all && hit.load() == times;
    }
    return all;
  }
} // namespace

// A task beside a loop: the caller runs it while the workers take chunks, then joins them; every index once and the
// task once, threaded, capped to one thread, with too little work for two, and nested in a busy pool. Its exception
// comes back after the loop's chunks have run.
TEST(parallel_for_beside_runs_the_task_and_every_index_once)
{
  beside_counts threaded {.hits = std::vector<std::atomic<int>>(257)};
  run_beside(threaded, std::size_t {1} << 20);
  EXPECT(every_index(threaded, 1, 1));
  beside_counts capped {.hits = std::vector<std::atomic<int>>(257)};
  numerical::set_max_threads(1);
  run_beside(capped, std::size_t {1} << 20);
  numerical::set_max_threads(0);
  EXPECT(every_index(capped, 1, 1));
  beside_counts light {.hits = std::vector<std::atomic<int>>(257)};
  run_beside(light, 1);
  EXPECT(every_index(light, 1, 1));
  beside_counts nested {.hits = std::vector<std::atomic<int>>(64)};
  numerical::detail::parallel_for(8, std::size_t {1} << 20, [&nested](std::size_t) { run_beside(nested, std::size_t {1} << 20); });
  EXPECT(every_index(nested, 8, 8));
  std::atomic<std::size_t> ran    {0};
  bool                     caught {false};
  try {
    numerical::detail::parallel_for_beside(1000, std::size_t {1} << 20, [&ran](std::size_t) { ++ran; },
                                           [] { throw std::runtime_error("beside"); });
  }
  catch (const std::runtime_error& error) {
    caught = std::string_view {error.what()} == "beside";
  }
  EXPECT(caught);
  EXPECT(ran.load() == 1000U);
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

// parallel_zero clears exactly its count, below the size that pays a second thread (one fill) and above it (chunks
// across the threads, the last one partial; past 16 MiB, by 2 MiB on Linux), and leaves the element past it alone.
TEST(parallel_zero_clears_its_count_and_nothing_past_it)
{
  using numerical::detail::parallel_zero;
  constexpr std::size_t threshold {2 * (numerical::detail::work_per_thread / numerical::detail::zero_work)};
  for (const std::size_t count : {std::size_t {0}, std::size_t {1000}, threshold - 1, threshold,
                                  threshold + (numerical::detail::zero_chunk / 2) + 3, (std::size_t {1} << 21U) + 77}) {
    std::vector<double> data(count + 1, 1.5);
    parallel_zero(data.data(), count);
    EXPECT(std::all_of(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(count), [](double x) { return x == 0.0; }));
    EXPECT_EQ(data[count], 1.5);
  }
}

// parallel_copy copies exactly its count on both sides of the size that pays a second thread and past 16 MiB, and
// leaves the element past it alone.
TEST(parallel_copy_copies_its_count_and_nothing_past_it)
{
  using numerical::detail::parallel_copy;
  constexpr std::size_t threshold {2 * (numerical::detail::work_per_thread / numerical::detail::zero_work)};
  for (const std::size_t count : {std::size_t {0}, std::size_t {1000}, threshold - 1, threshold,
                                  threshold + (numerical::detail::zero_chunk / 2) + 3, (std::size_t {1} << 21U) + 77}) {
    std::vector<double> in(count);
    for (std::size_t i {0}; i < count; ++i) {
      in[i] = static_cast<double>(i) * 0.5;
    }
    std::vector<double> out(count + 1, -1.0);
    parallel_copy(in.data(), count, out.data());
    EXPECT(std::equal(in.begin(), in.end(), out.begin()));
    EXPECT_EQ(out[count], -1.0);
  }
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
// One pool thread per physical core where the system tells them, else per hardware thread, at least one.
TEST(the_pool_runs_one_thread_per_physical_core)
{
  using numerical::detail::pool_threads;
  EXPECT_EQ(pool_threads(16, 8), 8U);  // hyperthreads: the cores
  EXPECT_EQ(pool_threads(16, 0), 16U); // the cores unknown: the hardware threads
  EXPECT_EQ(pool_threads(4, 8), 4U);   // never more than the hardware threads
  EXPECT_EQ(pool_threads(0, 0), 1U);   // an unknown hardware: one
  EXPECT(numerical::detail::hardware_threads() >= 1U);
}

// One worker per known core, pinned there, the one on the caller's core sitting the run out; with the cores
// unknown, a free worker per thread beside the caller, none sitting out.
TEST(the_pool_starts_a_worker_per_core_and_leaves_the_callers_out)
{
  using numerical::detail::no_core;
  using numerical::detail::sits_out;
  using numerical::detail::worker_cores;
  EXPECT(worker_cores(4, 4) == (std::vector<std::size_t> {0, 1, 2, 3}));
  EXPECT(worker_cores(0, 3) == (std::vector<std::size_t> {no_core, no_core}));
  EXPECT(worker_cores(0, 1).empty());
  EXPECT(sits_out(2, 2));
  EXPECT(!sits_out(2, 3));
  EXPECT(!sits_out(no_core, no_core));
}

#if defined(__linux__)
namespace {
  // A topology tree under a fresh directory: processor p's core and package as given, or a file left out or not a
  // number where a test says.
  struct fake_topology
  {
    std::filesystem::path root;

    explicit fake_topology(const std::string& name)
      : root(std::filesystem::temp_directory_path() / ("numerical-topology-" + name + "-" + std::to_string(getpid())))
    {
      std::filesystem::remove_all(root);
    }

    ~fake_topology()
    {
      std::error_code ignored;
      std::filesystem::remove_all(root, ignored);
    }

    fake_topology(const fake_topology&)            = delete;
    fake_topology& operator=(const fake_topology&) = delete;

    void processor(int                cpu,
                   const std::string& core,
                   const std::string& package) const
    {
      const std::filesystem::path dir {root / ("cpu" + std::to_string(cpu)) / "topology"};
      std::filesystem::create_directories(dir);
      std::ofstream {dir / "core_id"} << core << "\n";
      if (!package.empty()) {
        std::ofstream {dir / "physical_package_id"} << package << "\n";
      }
    }

    [[nodiscard]] std::vector<cpu_set_t> sets(std::initializer_list<int> allowed) const
    {
      cpu_set_t set;
      CPU_ZERO(&set); // NOLINT(hicpp-signed-bitwise): the system's macro
      for (const int cpu : allowed) {
        CPU_SET(cpu, &set); // NOLINT(hicpp-signed-bitwise): the system's macro
      }
      return numerical::detail::core_sets_from(set, root.string() + "/");
    }
  };

  [[nodiscard]] bool holds(const cpu_set_t&           set,
                           std::initializer_list<int> cpus)
  {
    int count {0};
    for (const int cpu : cpus) {
      count += CPU_ISSET(cpu, &set) != 0 ? 1 : 0; // NOLINT(hicpp-signed-bitwise): the system's macro
    }
    return count == static_cast<int>(cpus.size()) && CPU_COUNT(&set) == count; // NOLINT(hicpp-signed-bitwise)
  }
} // namespace

// A core's hyperthreads share a set, cores by their first processor, packages apart; a processor outside the
// affinity is not read; a topology missing or not a number makes no claim.
TEST(core_sets_read_the_topology_of_the_allowed_processors)
{
  const fake_topology tree {"sets"};
  tree.processor(0, "0", "0");
  tree.processor(1, "1", "0");
  tree.processor(2, "0", "0"); // processor 0's sibling
  tree.processor(3, "0", "1"); // core 0 of another package
  const std::vector<cpu_set_t> sets {tree.sets({0, 1, 2, 3})};
  EXPECT(sets.size() == 3 && holds(sets[0], {0, 2}) && holds(sets[1], {1}) && holds(sets[2], {3}));
  EXPECT(tree.sets({1}).size() == 1);
  EXPECT(tree.sets({}).empty());
  EXPECT(tree.sets({0, 5}).empty()); // processor 5 has no topology
  const fake_topology odd {"odd"};
  odd.processor(0, "zero", "0");
  odd.processor(1, "1", "");
  EXPECT(odd.sets({0}).empty()); // not a number
  EXPECT(odd.sets({1}).empty()); // no package
}

// Each core's processors once, the cores as many as physical_cores counts; a thread pinned to a core runs there
// alone, an index past the cores leaves it as it was, and every worker that takes a chunk is on one core.
TEST(the_pool_pins_each_worker_to_a_core_of_its_own)
{
  const std::vector<cpu_set_t> cores {numerical::detail::core_sets()};
  EXPECT_EQ(cores.size(), numerical::detail::physical_cores());
  cpu_set_t seen;
  CPU_ZERO(&seen); // NOLINT(hicpp-signed-bitwise): the system's macro
  bool disjoint {true};
  for (cpu_set_t core : cores) {
    cpu_set_t both;
    CPU_AND(&both, &seen, &core);                 // NOLINT(hicpp-signed-bitwise): the system's macro
    disjoint = disjoint && CPU_COUNT(&both) == 0; // NOLINT(hicpp-signed-bitwise): the system's macro
    CPU_OR(&seen, &seen, &core);                  // NOLINT(hicpp-signed-bitwise): the system's macro
  }
  EXPECT(disjoint);
  const std::vector<std::size_t> table  {numerical::detail::processor_cores(cores)};
  bool                           mapped {true};
  for (std::size_t c {0}; c < cores.size(); ++c) {
    for (int cpu {0}; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &cores[c]) != 0) { // NOLINT(hicpp-signed-bitwise): the system's macro
        mapped = mapped && numerical::detail::core_of(table, cpu) == c;
      }
    }
  }
  EXPECT(mapped);
  EXPECT_EQ(numerical::detail::core_of(table, -1), numerical::detail::no_core);
  EXPECT_EQ(numerical::detail::core_of(table, CPU_SETSIZE), numerical::detail::no_core);
  const auto affinity {[] {
                         cpu_set_t set;
                         CPU_ZERO(&set); // NOLINT(hicpp-signed-bitwise): the system's macro
                         (void)sched_getaffinity(0, sizeof(set), &set);
                         return set;
                       }};
  const auto one_of_the_cores {[&cores](const cpu_set_t& set) {
                                 return std::any_of(cores.begin(), cores.end(), [&set](const cpu_set_t& core) { return CPU_EQUAL(&set, &core) != 0; }); // NOLINT(hicpp-signed-bitwise): the system's macro
                               }};
  const cpu_set_t before {affinity()};
  bool            left   {false};
  bool            pinned {false};
  std::thread {[&] {
                 numerical::detail::pin_to_core(cores, cores.size());
                 const cpu_set_t after {affinity()};
                 left = CPU_EQUAL(&before, &after) != 0; // NOLINT(hicpp-signed-bitwise): the system's macro
                 numerical::detail::pin_to_core(cores, 0);
                 pinned = cores.empty() || one_of_the_cores(affinity());
               }}.join();
  EXPECT(left);
  EXPECT(pinned);
  const std::thread::id  caller {std::this_thread::get_id()};
  std::mutex             guard;
  std::vector<cpu_set_t> workers;
  numerical::detail::parallel_for(4096, std::size_t {1} << 20, [&](std::size_t) {
                                    if (std::this_thread::get_id() != caller) {
                                      const cpu_set_t                   set {affinity()};
                                      const std::lock_guard<std::mutex> held {guard};
                                      workers.push_back(set);
                                    }
                                  });
  EXPECT(cores.empty() || std::all_of(workers.begin(), workers.end(), one_of_the_cores));
}
#endif

// A run rings a worker by rank among the cores, the caller's left out; a worker free to move always joins.
TEST(a_run_wants_the_workers_by_rank_without_the_callers)
{
  using numerical::detail::no_core;
  struct asked
  {
    std::size_t core;
    std::size_t caller;
    std::size_t helpers;
    bool        joins;
  };
  // Core 3 is rank 2 beside a caller on core 2; with no caller's core known, a worker's rank is its core.
  const std::vector<asked> cases {
    {.core = 2, .caller = 2, .helpers = 5, .joins = false},       {.core = 1, .caller = 2, .helpers = 2, .joins = true},
    {.core = 3, .caller = 2, .helpers = 2, .joins = false},       {.core = 3, .caller = 2, .helpers = 3, .joins = true},
    {.core = 0, .caller = no_core, .helpers = 1, .joins = true},  {.core = 1, .caller = no_core, .helpers = 1, .joins = false},
    {.core = no_core, .caller = 0, .helpers = 0, .joins = true}};
  for (const asked& run : cases) {
    EXPECT_EQ(numerical::detail::wants(run.core, run.caller, run.helpers), run.joins);
  }
}

// Workers asleep (the polling long over) are rung by the next run: chunk 0 waits for chunk 1 to start on another
// thread, which only a woken worker can take, the caller being held in chunk 0.
TEST(a_run_wakes_the_workers_asleep)
{
  if (numerical::max_threads() < 2) {
    return;
  }
  for (int round {0}; round < 3; ++round) {
    std::this_thread::sleep_for(numerical::detail::spin_time * 4);
    std::atomic<bool> second {false};
    std::atomic<bool> met    {false};
    numerical::detail::parallel_for(2, std::size_t {1} << 20, [&second, &met](std::size_t i) {
                                      if (i == 1) {
                                        second = true;
                                        return;
                                      }
                                      const auto until {std::chrono::steady_clock::now() + std::chrono::seconds {2}};
                                      while (!second.load() && std::chrono::steady_clock::now() < until) {
                                        std::this_thread::yield();
                                      }
                                      met = second.load();
                                    });
    EXPECT(met.load());
  }
}

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
        end_child(2);
      }
      // On the stack: an allocator need not be safe to call in a child forked among threads (the address
      // sanitizer's is not), and the kernel itself allocates nothing here.
      std::array<std::atomic<int>, 512> hits {};
      numerical::detail::parallel_for(hits.size(), std::size_t {1} << 20, [&hits](std::size_t i) { ++hits[i]; });
      int total                              {0};
      for (const std::atomic<int>& hit : hits) {
        total += hit.load();
      }
      end_child(total == 512 ? 0 : 1);
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
