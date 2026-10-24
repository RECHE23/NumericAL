/*!
 * \file parallel.hpp
 * \brief A minimal `std::thread` parallel-for for the hot kernels.
 *
 * The only standard-library parallelism a header-only, zero-dependency leaf can
 * use portably: `std::thread`, no TBB, no execution policies, started once in a shared pool. \ref
 * numerical::detail::parallel_for splits a counted range across the hardware
 * threads — each index is handled by exactly one thread, so a body that writes a
 * disjoint slice of the output per index stays race-free **and deterministic**
 * (the result is identical to the sequential run, bit for bit). The caller states the
 * work of one index; a range runs on as many threads as its work pays for, and
 * sequentially when that is fewer than two, since waking a thread and handing it work costs time too.
 */
#ifndef NUMERICAL_PARALLEL_HPP
#define NUMERICAL_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
#else
#  include <unistd.h>
#endif
#if defined(__linux__)
#  include <fstream>
#  include <map>
#  include <string>
#  include <utility>
#  include <sched.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

namespace numerical::detail {

  //! \brief This process's identifier: a pool started before a `fork` belongs to the parent, whose threads the
  //!        child does not have.
  [[nodiscard]] inline long current_process() noexcept
  {
#if defined(_WIN32)
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
  }

  //! \brief No core: a worker free to run anywhere, or a caller on a processor the pool does not know.
  inline constexpr std::size_t no_core {std::numeric_limits<std::size_t>::max()};

  /*!
   * \brief The core of each worker the pool starts beside its callers, from the \p cores known (0: none) and its
   *        \p threads (at least 1): one worker per core, pinned there, where the cores are known (each run then
   *        leaves out the one on the caller's core, \ref sits_out); else `threads − 1` workers free to move.
   */
  [[nodiscard]] inline std::vector<std::size_t> worker_cores(std::size_t cores,
                                                             std::size_t threads)
  {
    std::vector<std::size_t> out(cores != 0 ? cores : threads - 1, no_core);
    for (std::size_t c {0}; c < cores; ++c) {
      out[c] = c;
    }
    return out;
  }

  //! \brief Whether the worker pinned to core \p worker sits out a run whose caller is on core \p caller: one thread
  //!        per core, as two on one core's hyperthreads share its multiply-add units. A free worker never does.
  [[nodiscard]] constexpr bool sits_out(std::size_t worker,
                                        std::size_t caller) noexcept
  {
    return worker != no_core && worker == caller;
  }

#if defined(__linux__)
  //! \brief The whole first line of \p path, or nothing where it cannot be read.
  [[nodiscard]] inline std::string first_line(const std::string& path)
  {
    std::ifstream file {path};
    std::string   line;
    std::getline(file, line);
    return line;
  }

  /*!
   * \brief The processors of each physical core among \p allowed, one set per distinct (package, core) pair, in the
   *        order of each one's first processor, from the topology under \p cpu_root (`/sys/devices/system/cpu/` on
   *        Linux); none where a processor's topology is missing or not a number (\c std::stol throws) or memory
   *        runs out: no claim.
   */
  [[nodiscard]] inline std::vector<cpu_set_t> core_sets_from(const cpu_set_t&   allowed,
                                                             const std::string& cpu_root) noexcept
  {
    try {
      std::map<std::pair<long, long>, std::size_t> index; // (package, core) → its set
      std::vector<cpu_set_t>                       sets;
      for (int cpu {0}; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed) == 0) { // NOLINT(hicpp-signed-bitwise): the system's macro
          continue;
        }
        const std::string topology {cpu_root + "cpu" + std::to_string(cpu) + "/topology/"};
        const long        core     {std::stol(first_line(topology + "core_id"))};
        const long        package  {std::stol(first_line(topology + "physical_package_id"))};
        const auto [where, added] {index.try_emplace({package, core}, sets.size())};
        if (added) {
          sets.emplace_back();
          CPU_ZERO(&sets.back()); // NOLINT(hicpp-signed-bitwise): the system's macro
        }
        CPU_SET(cpu, &sets[where->second]); // NOLINT(hicpp-signed-bitwise): the system's macro
      }
      return sets;
    }
    catch (...) {
      return {};
    }
  }

  /*!
   * \brief The processors of each physical core this process may run on (\ref core_sets_from over its affinity and
   *        `/sys`); none where the topology is unknown. An affinity the system does not give stays empty: no core.
   */
  [[nodiscard]] inline std::vector<cpu_set_t> core_sets() noexcept
  {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);                                                 // NOLINT(hicpp-signed-bitwise): the system's macro
    static_cast<void>(sched_getaffinity(0, sizeof(allowed), &allowed)); // on failure the set stays empty
    return core_sets_from(allowed, "/sys/devices/system/cpu/");
  }

  //! \brief The core index of each processor of \p cores (\ref no_core for the others), by processor number.
  [[nodiscard]] inline std::vector<std::size_t> processor_cores(const std::vector<cpu_set_t>& cores)
  {
    std::vector<std::size_t> table(CPU_SETSIZE, no_core);
    for (std::size_t c {0}; c < cores.size(); ++c) {
      for (std::size_t cpu {0}; cpu < table.size(); ++cpu) {
        if (CPU_ISSET(cpu, &cores[c]) != 0) { // NOLINT(hicpp-signed-bitwise): the system's macro
          table[cpu] = c;
        }
      }
    }
    return table;
  }

  //! \brief The core of processor \p cpu in \p table (`processor_cores`), or \ref no_core: a negative \p cpu is
  //!        `sched_getcpu`'s failure.
  [[nodiscard]] inline std::size_t core_of(const std::vector<std::size_t>& table,
                                           int                             cpu) noexcept
  {
    // Through the subspan from `cpu`: GCC 15 at -O3 reads `table[cpu]` behind an index check (a conditional or an
    // early return) before the check, and refuses the subscript past the end (-Werror=array-bounds).
    const std::span<const std::size_t> rest {std::span {table}.subspan(cpu < 0 ? table.size() : std::min(table.size(), static_cast<std::size_t>(cpu)))};
    return rest.empty() ? no_core : rest.front();
  }

  //! \brief Pins the calling thread to the processors of `cores[index]`, where there is one; else, or should the
  //!        system refuse, leaves it where it may run.
  inline void pin_to_core(const std::vector<cpu_set_t>& cores,
                          std::size_t                   index) noexcept
  {
    if (index < cores.size()) {
      (void)sched_setaffinity(0, sizeof(cpu_set_t), &cores[index]);
    }
  }

#endif

  /*!
   * \brief The physical cores this process may run on, or 0 where the system does not say. Linux: `core_sets`;
   *        macOS: `hw.physicalcpu`.
   */
  [[nodiscard]] inline std::size_t physical_cores() noexcept
  {
#if defined(__linux__)
    return core_sets().size();
#elif defined(__APPLE__)
    int         cores {0};
    std::size_t size  {sizeof(cores)};
    (void)sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0); // on failure cores stays 0
    return static_cast<std::size_t>(std::max(cores, 0));
#else
    return 0;
#endif
  }

  //! \brief The pool's threads from the \p logical hardware threads, the \p physical cores and the processors a CPU
  //!        \p quota allows (each 0: unknown, or none): the cores where known, no more than the quota, at least 1.
  [[nodiscard]] constexpr std::size_t pool_threads(std::size_t logical,
                                                   std::size_t physical,
                                                   std::size_t quota = 0) noexcept
  {
    const std::size_t hardware {std::max<std::size_t>(1, logical)};
    const std::size_t cores    {physical == 0 ? hardware : std::min(physical, hardware)};
    return quota == 0 ? cores : std::max<std::size_t>(1, std::min(cores, quota));
  }

  //! \brief The processors a CPU quota of \p quota microseconds a \p period allows, rounded up (cgroup v1's
  //!        `cpu.cfs_quota_us`, -1 for none, and `cpu.cfs_period_us`): 0 for none.
  [[nodiscard]] constexpr std::size_t quota_processors(long long quota,
                                                       long long period) noexcept
  {
    return quota <= 0 || period <= 0 ? 0 : static_cast<std::size_t>((quota + period - 1) / period);
  }

  //! \brief \ref quota_processors of cgroup v2's `cpu.max` text, "<quota> <period>" ("max" for none): 0 for none or a
  //!        text it does not read.
  [[nodiscard]] inline std::size_t cpu_max_processors(std::string_view text) noexcept
  {
    const auto number {[](std::string_view digits, long long& value) {
                         const auto [end, error] {std::from_chars(digits.data(), digits.data() + digits.size(), value)};
                         return error == std::errc {} && end == digits.data() + digits.size();
                       }};
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) {
      text.remove_suffix(1);
    }
    const std::size_t space  {text.find(' ')};
    long long         quota  {0};
    long long         period {0};
    if (space == std::string_view::npos || !number(text.substr(0, space), quota) || !number(text.substr(space + 1), period)) {
      return 0;
    }
    return quota_processors(quota, period);
  }

#if defined(__linux__)
  /*!
   * \brief The processors the CPU quotas over this process allow, the least of them, 0 where there is none: a container
   *        started with two processors' quota on a machine of many cores, whose quota a pool of a thread a core would
   *        exhaust and then wait out (six AVX2 cores, a quota of two: products of order 128 took 100 µs on six threads,
   *        59 on two). cgroup v2: `cpu.max` of this process's group (`/proc/self/cgroup`'s `0::` line) and of each
   *        group above it; cgroup v1: `cpu.cfs_quota_us` over `cpu.cfs_period_us`.
   */
  [[nodiscard]] inline std::size_t cgroup_quota()
  {
    std::size_t   least {0};
    const auto    keep {[&least](std::size_t processors) {
                          if (processors != 0 && (least == 0 || processors < least)) {
                            least = processors;
                          }
                        }};
    std::ifstream self {"/proc/self/cgroup"};
    std::string   line;
    std::string   group;
    while (std::getline(self, line)) {
      if (line.rfind("0::", 0) == 0) {
        group = line.substr(3);
      }
    }
    while (!group.empty() && group != "/") {
      keep(cpu_max_processors(first_line("/sys/fs/cgroup" + group + "/cpu.max")));
      group.erase(group.rfind('/'));
    }
    keep(cpu_max_processors(first_line("/sys/fs/cgroup/cpu.max")));
    for (const char* const v1 : {"/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"}) {
      const std::string quota  {first_line(std::string {v1} + "/cpu.cfs_quota_us")};
      const std::string period {first_line(std::string {v1} + "/cpu.cfs_period_us")};
      keep(cpu_max_processors(quota + " " + period));
    }
    return least;
  }

  //! \brief \ref cgroup_quota, 0 where reading fails.
  [[nodiscard]] inline std::size_t cgroup_processors() noexcept
  {
    try {
      return cgroup_quota();
    }
    catch (...) { // a file or a string that cannot be had: no quota known
      return 0;
    }
  }

#endif

  //! \brief The processors the CPU quotas over this process allow (`cgroup_processors` on Linux), 0 for none.
  [[nodiscard]] inline std::size_t quota_limit() noexcept
  {
#if defined(__linux__)
    return cgroup_processors();
#else
    return 0;
#endif
  }

  /*!
   * \brief The threads the pool runs, the caller's included (at least 1), known without starting the pool: one per
   *        physical core where the system tells them (else one per hardware thread), no more than a CPU quota allows
   *        (\ref quota_limit). Two threads on one core's
   *        hyperthreads share its multiply-add units: a product of order 128 on eight AVX2 cores with sixteen
   *        hyperthreads took 140 µs on sixteen threads, 71 on eight; order 2048, the same either way. Asked once,
   *        as the pool's workers are started once (glibc reads a file for the hardware threads on every call).
   */
  [[nodiscard]] inline std::size_t hardware_threads() noexcept
  {
    static const std::size_t threads {pool_threads(std::thread::hardware_concurrency(), physical_cores(), quota_limit())};
    return threads;
  }

  //! \brief How long a thread polls before it sleeps: kernels run in quick succession (a matrix product packs, then
  //!        computes, panel after panel), and a thread woken from sleep starts tens of microseconds late, its wake-up
  //!        paid by the caller of the next run, about 10 µs a sleeping worker on six x86-64 cores. A factorisation's
  //!        steps last about 100 µs at order 256, so at 100 µs the workers one step left out were asleep for the next
  //!        (an LU of order 256: six x86-64 cores 560 → 500 µs at 500; eight arm64 cores 358 → 324).
  inline constexpr std::chrono::microseconds spin_time {500};

  //! \brief Polls \p ready, yielding the core between polls, until it holds or \ref spin_time has passed.
  template <typename Ready>
  void spin_until(const Ready& ready)
  {
    const std::chrono::steady_clock::time_point until {std::chrono::steady_clock::now() + spin_time};
    while (!ready() && std::chrono::steady_clock::now() < until) {
      std::this_thread::yield();
    }
  }

  //! \brief A chunk of a parallel_for, type-erased: runs `body(i)` for `i` in `[begin, end)`.
  using chunk_runner = void (*)(const void* body, std::size_t begin, std::size_t end);

  //! \brief A task the caller of a run does before it takes chunks (\ref parallel_for_beside), type-erased.
  using side_runner = void (*)(const void* side);

  /*!
   * \brief One parallel_for handed to the pool: its chunks, the next one to take, and how many are done. Each
   *        run has its own, so a worker still leaving one run never touches the next.
   */
  struct pool_job
  {
    chunk_runner                    run      {};        //!< The chunk runner.
    const void*                     body     {};        //!< The body it is handed.
    std::size_t                     count    {};        //!< The number of iterations.
    std::size_t                     chunk    {};        //!< Indices per chunk.
    std::size_t                     chunks   {};        //!< Chunks.
    std::size_t                     helpers  {};        //!< The workers that may join it beside the caller.
    std::size_t                     core     {no_core}; //!< The caller's core: its worker sits it out (\ref sits_out).
    std::atomic<std::size_t>        joined   {0};       //!< The workers that asked to.
    std::atomic<std::size_t>        next     {0};       //!< The next chunk to take.
    std::atomic<std::size_t>        finished {0};       //!< The chunks done.
    std::mutex                      lock;               //!< Orders the last chunk's signal against the caller's sleep.
    std::condition_variable         done;               //!< Signalled when the last chunk finishes.
    std::vector<std::exception_ptr> failed;             //!< The exception of each chunk, if it threw.

    //! \brief Takes chunks until none is left, keeping the exception each throws.
    void work() noexcept
    {
      for (std::size_t c {next.fetch_add(1)}; c < chunks; c = next.fetch_add(1)) {
        try {
          run(body, c * chunk, std::min(count, (c + 1) * chunk));
        }
        catch (...) {
          failed[c] = std::current_exception();
        }
        if (finished.fetch_add(1) + 1 == chunks) {
          const std::lock_guard<std::mutex> held {lock};
          done.notify_all();
        }
      }
    }
  };

  //! \brief A worker's own wake-up: its core, whether it sleeps, and what wakes it.
  struct worker_bell
  {
    explicit worker_bell(std::size_t on) : core {on}
    {}

    const std::size_t       core;             //!< The core it is pinned to, or no_core.
    std::atomic<bool>       asleep {false};   //!< Set before it checks for a run and sleeps.
    std::mutex              lock;             //!< Orders a ring against its sleep.
    std::condition_variable rung;             //!< Signalled when a run that wants it starts.
  };

  //! \brief Whether the worker on \p core joins a run of \p helpers workers whose caller is on \p caller: by rank
  //!        among the cores, the caller's aside, so a run of few threads takes the same cores from one run to the next.
  [[nodiscard]] constexpr bool wants(std::size_t core,
                                     std::size_t caller,
                                     std::size_t helpers) noexcept
  {
    if (core == no_core) {
      return true;
    }
    if (core == caller) {
      return false;
    }
    const std::size_t rank {caller == no_core || core < caller ? core : core - 1};
    return rank < helpers;
  }

  /*!
   * \brief The threads every kernel shares, started once: a parallel_for costs a wake-up, not a thread start.
   *
   * The caller takes chunks beside the workers, each taking the next chunk as it frees up (a slower core takes
   * fewer). One run at a time: a parallel_for while the pool is busy (another thread's, or one nested in a
   * body) runs its chunks itself, with the same result. The pool is never destroyed: its threads wait for work
   * until the process ends, so no static destructor races them.
   */
  class thread_pool
  {
  public:

    thread_pool(const thread_pool&)            = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&)                 = delete;
    thread_pool& operator=(thread_pool&&)      = delete;
    ~thread_pool()                             = delete;

    /*!
     * \brief The pool, started on first use with a worker per hardware thread beside the caller.
     * \throws std::system_error If the system starts no thread (the workers already started keep the pool,
     *         which a later call starts again).
     */
    static thread_pool& shared()
    {
      static thread_pool* const pool {[] {
                                        auto* const started {new thread_pool {}};
                                        started->start();
                                        return started;
                                      }()};
      return *pool;
    }

    /*!
     * \brief Runs \p chunks chunks of \p chunk indices up to \p count on the caller and at most \p helpers workers,
     *        then rethrows the first exception in index order. \return false, having run nothing, when the pool is
     *        busy or belongs to the parent of a forked process. With \p side, the caller runs it first, while the
     *        workers take chunks, and its exception comes back before theirs.
     */
    bool try_run(chunk_runner run,
                 const void*  body,
                 std::size_t  count,
                 std::size_t  chunk,
                 std::size_t  chunks,
                 std::size_t  helpers,
                 side_runner  side      = nullptr,
                 const void*  side_body = nullptr)
    {
      bool idle {false};
      // A forked child has none of the parent's workers, and their locks may be held for good: it runs alone.
      if (owner_ != current_process() || !busy_.compare_exchange_strong(idle, true)) {
        return false;
      }
      struct release
      {
        std::atomic<bool>& busy;

        explicit release(std::atomic<bool>& flag) : busy {flag}
        {}

        release(const release&)            = delete;
        release& operator=(const release&) = delete;
        release(release&&)                 = delete;
        release& operator=(release&&)      = delete;
        ~release()
        {
          busy.store(false);
        }
      } const freed  {busy_};
      const auto job {std::make_shared<pool_job>()};
      job->run     = run;
      job->body    = body;
      job->count   = count;
      job->chunk   = chunk;
      job->chunks  = chunks;
      job->helpers = helpers;
#if defined(__linux__)
      job->core = core_of(processor_cores_, sched_getcpu());
#endif
      job->failed.resize(chunks);
      {
        const std::lock_guard<std::mutex> held {lock_};
        current_ = job;
        generation_.fetch_add(1);
      }
      for (std::size_t w {0}; w < bells_.size(); ++w) { // the workers the run wants, asleep: not the one on the caller's core
        worker_bell& bell {*bells_[w]};
#if defined(__linux__)                                  // elsewhere no worker is pinned, and every one may join
        if (!wants(bell.core, job->core, helpers)) {
          continue;
        }
#endif
        if (bell.asleep.load()) {
          { const std::lock_guard<std::mutex> held {bell.lock}; }
          bell.rung.notify_one();
        }
      }
      std::exception_ptr side_failed;
      if (side != nullptr) {
        try {
          side(side_body);
        }
        catch (...) {
          side_failed = std::current_exception();
        }
      }
      job->work();
      const auto all_done {[&job] { return job->finished.load() == job->chunks; }};
      spin_until(all_done);
      {
        std::unique_lock<std::mutex> held {job->lock};
        job->done.wait(held, all_done);
      }
      if (side_failed) {
        std::rethrow_exception(side_failed);
      }
      for (const std::exception_ptr& error : job->failed) {
        if (error) {
          std::rethrow_exception(error);
        }
      }
      return true;
    }

  private:

    thread_pool() = default;

    /*!
     * \brief Starts the workers (after construction: each uses the pool), `worker_cores`: on Linux one per core,
     *        pinned there. Free to move, as a caller is, two threads at times shared one core's hyperthreads (a
     *        product of order 1024 on eight AVX2 cores took 8.1 ms, or 10.4 in most processes).
     */
    void start()
    {
#if defined(__linux__)
      const std::vector<cpu_set_t> cores {core_sets()};
      processor_cores_ = processor_cores(cores);
      for (const std::size_t core : worker_cores(cores.size(), hardware_threads())) {
        bells_.push_back(std::make_unique<worker_bell>(core));
        worker_bell* const bell {bells_.back().get()};
        std::thread {[this, cores, core, bell] {
                       pin_to_core(cores, core);
                       serve(*bell);
                     }}.detach();
        ++workers_;
      }
#else
      for (const std::size_t core : worker_cores(0, hardware_threads())) {
        bells_.push_back(std::make_unique<worker_bell>(core));
        worker_bell* const bell {bells_.back().get()};
        std::thread {[this, bell] { serve(*bell); }}.detach();
        ++workers_;
      }
#endif
    }

    //! \brief A worker on \p core: waits for a run (polling a while, then asleep), works on it unless it sits it
    //!        out, waits for the next.
    [[noreturn]] void serve(worker_bell& bell)
    {
      [[maybe_unused]] const std::size_t core {bell.core};
      std::size_t                        seen {0};
#if defined(__linux__)
      bool spin                               {true}; // the worker that sat a run out sleeps till the next: polling, it slowed its core's caller
#endif
      for (;;) {
        const auto started {[this, &seen] { return generation_.load() != seen; }};
#if defined(__linux__)
        if (spin) {
          spin_until(started);
        }
#else
        spin_until(started);
#endif
        if (!started()) { // asleep till a run that wants it rings: one that leaves it out never wakes it
          bell.asleep.store(true);
          std::unique_lock<std::mutex> held {bell.lock};
          bell.rung.wait(held, started);
          bell.asleep.store(false);
        }
        std::shared_ptr<pool_job> job;
        {
          const std::lock_guard<std::mutex> held {lock_};
          seen = generation_.load();
          job  = current_;
        }
#if defined(__linux__) // elsewhere no worker is pinned, so none sits out
        spin = !sits_out(core, job->core);
        if (!spin) {
          continue;
        }
#endif
#if defined(__linux__)
        // By rank among the cores, the caller's aside, not by arrival: a run of few threads then takes the same cores
        // from one run to the next, their caches still warm (ten AVX-512 cores, an LU of order 1024 on two threads:
        // 21.5 → 14.6 ms, on four 18 → 11.4; six AVX2 cores unchanged).
        const bool        wanted {wants(core, job->core, job->helpers)};
#else
        const bool wanted        {true};
#endif
        if (wanted && job->joined.fetch_add(1) < job->helpers) { // as many workers as the run's threads, the caller aside
          job->work();
        }
      }
    }

    std::mutex                                lock_;               //!< Guards current_ and generation_.
    std::vector<std::unique_ptr<worker_bell>> bells_;              //!< Each worker's, set before it starts.
    std::shared_ptr<pool_job>                 current_;            //!< The latest run.
    std::atomic<std::size_t>                  generation_ {0};     //!< Counts the runs (polled without the lock).
    std::atomic<bool>                         busy_       {false}; //!< Whether a run holds the pool.
    std::size_t                               workers_    {0};     //!< The threads started.
#if defined(__linux__)
    std::vector<std::size_t> processor_cores_;                     //!< Each processor's core (`processor_cores`), set before the workers start.
#endif
    long                      owner_      {current_process()};     //!< The process whose threads they are.
  };

  //! \brief The most threads a parallel_for uses (0: as many as the pool has), set by \ref numerical::set_max_threads.
  [[nodiscard]] inline std::atomic<std::size_t>& thread_cap() noexcept
  {
    static std::atomic<std::size_t> cap {0};
    return cap;
  }

  //! \brief The work one thread must have to pay for starting it: about 60 µs of multiply-adds, six times
  //!        the cost of a start (measured on arm64: 16 indices of 4096 ran 80 µs sequentially, 92 µs on
  //!        eight threads; of 16384, 324 µs and 128 µs).
  inline constexpr std::size_t work_per_thread {std::size_t {1} << 16};

  //! \brief The elements one task of an element-wise operation takes, a quarter of a thread's work: a tensor of a
  //!        million shares out (M1 Pro, 2000 × 2000 doubles added: 3.1 → 1.4 ms), one of a few hundred thousand stays
  //!        within its time (a sixteenth: 300 × 300 plus a row 13 % slower, each task setting up its odometer).
  inline constexpr std::size_t elementwise_chunk {work_per_thread / 4};

  //! \brief The multiply-adds a vectorised inner loop does in the time of one in \ref work_per_thread's
  //!        dependent chain: a product weighs its work by it (at order 64, a matrix product on eight threads
  //!        took 95 µs, on one 35 µs).
  inline constexpr std::size_t vector_speedup {4};

  /*!
   * \brief How many threads \p count indices of \p work_per_index each pay for: one per
   *        \ref work_per_thread, at most one per index and per hardware thread.
   * \param[in] count          The number of indices.
   * \param[in] work_per_index The work of one index, in multiply-adds.
   * \param[in] hardware       The hardware threads.
   */
  [[nodiscard]] inline std::size_t threads_for(std::size_t count,
                                               std::size_t work_per_index,
                                               std::size_t hardware)
  {
    const std::size_t total {work_per_index != 0 && count > std::numeric_limits<std::size_t>::max() / work_per_index
                             ? std::numeric_limits<std::size_t>::max()
                             : count * work_per_index};
    return std::min({total / work_per_thread, count, std::max<std::size_t>(1, hardware)});
  }

  /*!
   * \brief The threads \ref parallel_for shares \p count indices of \p work_per_index out to (\ref threads_for): the
   *        hardware's, at most the cap \ref numerical::set_max_threads set. Fewer than two runs them on the caller.
   */
  [[nodiscard]] inline std::size_t parallel_threads(std::size_t count,
                                                    std::size_t work_per_index)
  {
    const std::size_t cap {thread_cap().load()};
    return threads_for(count, work_per_index, cap == 0 ? hardware_threads() : std::min(cap, hardware_threads()));
  }

  /*!
   * \brief Runs `body(i)` for every `i` in `[0, count)`.
   *
   * The range is partitioned across as many threads as its work pays for (\ref threads_for), and runs
   * sequentially when that is fewer than two. \p body must be safe to invoke concurrently for distinct `i`
   * (it is when each call touches a disjoint part of the output).
   *
   * An exception \p body throws in a worker is kept and rethrown here once every
   * worker has joined (the first chunk's, in index order): one escaping a worker
   * thread would call `std::terminate`. A worker the system cannot start is a
   * `std::system_error`, thrown once the others have joined.
   *
   * \param[in] count          The number of iterations.
   * \param[in] work_per_index The work of one iteration, in multiply-adds.
   * \param[in] body           The per-index work (`void(std::size_t)`).
   */
  template <typename Body>
  void parallel_for(std::size_t count,
                    std::size_t work_per_index,
                    const Body& body)
  {
    const std::size_t threads {parallel_threads(count, work_per_index)};
    const auto        runner  {[](const void* erased, std::size_t begin, std::size_t end) {
                                 const Body& work {*static_cast<const Body*>(erased)};
                                 for (std::size_t i {begin}; i < end; ++i) {
                                   work(i);
                                 }
                               }};
    if (threads < 2) {
      runner(&body, 0, count);
      return;
    }
    // Up to four chunks a thread, taken as threads free up, so a slower core holds back no one.
    const std::size_t chunks {std::max<std::size_t>(1, std::min(count, threads * 4))};
    const std::size_t chunk  {((count + chunks) - 1) / chunks}; // a last chunk past the count runs nothing
    if (!thread_pool::shared().try_run(runner, &body, count, chunk, chunks, threads - 1)) {
      runner(&body, 0, count);
    }
  }

  /*!
   * \brief Runs \p side on the caller while the pool's workers run `body(i)` for every `i` in `[0, count)`, then lets
   *        the caller take the chunks left: one task beside a parallel loop, each touching what the other does not
   *        (a factorisation's next panel beside its trailing update). \p side's own parallel loops find the pool
   *        busy and run in its thread. Where the work pays for fewer than two threads, or the pool is busy, \p side
   *        runs, then the loop, on the caller. An exception of \p side comes back after the loop's chunks have
   *        joined, before theirs.
   * \param[in] count          The number of iterations.
   * \param[in] work_per_index The work of one iteration, in multiply-adds.
   * \param[in] body           The per-index work (`void(std::size_t)`).
   * \param[in] side           The caller's task (`void()`).
   */
  template <typename Body,
            typename Side>
  void parallel_for_beside(std::size_t count,
                           std::size_t work_per_index,
                           const Body& body,
                           const Side& side)
  {
    const std::size_t cap     {thread_cap().load()};
    const std::size_t threads {threads_for(count, work_per_index, cap == 0 ? hardware_threads() : std::min(cap, hardware_threads()))};
    const auto        runner  {[](const void* erased, std::size_t begin, std::size_t end) {
                                 const Body& work {*static_cast<const Body*>(erased)};
                                 for (std::size_t i {begin}; i < end; ++i) {
                                   work(i);
                                 }
                               }};
    const auto beside {[](const void* erased) { (*static_cast<const Side*>(erased))(); }};
    if (threads >= 2) {
      const std::size_t chunks {std::max<std::size_t>(1, std::min(count, threads * 4))};
      const std::size_t chunk  {((count + chunks) - 1) / chunks};
      if (thread_pool::shared().try_run(runner, &body, count, chunk, chunks, threads - 1, beside, &side)) {
        return;
      }
    }
    side();
    runner(&body, 0, count);
  }

  //! \brief The elements one index of \ref parallel_zero writes: eight 4 KiB pages of `double`.
  inline constexpr std::size_t zero_chunk {std::size_t {1} << 12};

  /*!
   * \brief The elements one index of \ref parallel_zero or \ref parallel_copy takes of \p count: \ref zero_chunk, or,
   *        from the size at which a \ref buffer takes Linux's 2 MiB pages (16 MiB), one such page's worth, so that no
   *        two threads fault in the same page: they waited on each other's fault there (six x86-64 AVX2 cores, 32 MiB
   *        zeroed into fresh memory 3.6 → 2.5 ms; the LU and Cholesky of order 2000, their operand copied by eight 4 KiB
   *        pages a thread, took 2 to 4 % longer than with one thread's copy).
   */
  template <typename T>
  [[nodiscard]] constexpr std::size_t fault_chunk(std::size_t count) noexcept
  {
#if defined(__linux__)
    constexpr std::size_t huge_page  {std::size_t {1} << 21U};
    constexpr std::size_t huge_least {std::size_t {1} << 24U};
    if (count >= huge_least / sizeof(T)) {
      return huge_page / sizeof(T);
    }
#else
    (void)count;
#endif
    return zero_chunk;
  }

  //! \brief The work of zeroing one element, in multiply-adds: a second thread from 1 MiB of `double`. Below, the
  //!        allocator mostly hands back memory already touched, and zeroing 512 KiB took 4.5 µs on one arm64 thread,
  //!        9 µs split across eight.
  inline constexpr std::size_t zero_work {1};

  /*!
   * \brief Sets the \p count elements at \p out to zero, split across the threads the work pays for: a fresh
   *        result's pages are first touched, and faulted in, by several threads, where a sequential fill pays every
   *        fault on one (a product on eight arm64 threads: 16 % faster at order 512, 7 % at 1024). The zero is a
   *        constant in each chunk's fill, which the compiler turns into the C library's zeroing (on arm64 it clears
   *        whole cache lines without reading them; a value passed in, stored element by element, was 4 % slower at
   *        order 1024 on one thread).
   */
  template <typename T>
  void parallel_zero(T*          out,
                     std::size_t count)
  {
    if (count < 2 * (work_per_thread / zero_work)) { // too little to pay for a second thread
      std::fill(out, out + count, T {});
      return;
    }
    const std::size_t piece {fault_chunk<T>(count)};
    parallel_for((count + piece - 1) / piece, piece * zero_work, [out, count, piece](std::size_t chunk) {
                   const std::size_t begin {chunk * piece};
                   std::fill(out + begin, out + std::min(count, begin + piece), T {});
                 });
  }

  /*!
   * \brief Copies the \p count elements at \p in to \p out, split across the threads the work pays for, as
   *        \ref parallel_zero does: a fresh copy's pages are faulted in by several threads, where a sequential copy pays
   *        every fault on one (a 2 MiB copy into fresh memory took 0.85 ms on one x86-64 core in a container, a
   *        third of the Cholesky factorisation of order 512).
   */
  template <typename T>
  void parallel_copy(const T*    in,
                     std::size_t count,
                     T*          out)
  {
    if (count < 2 * (work_per_thread / zero_work)) { // too little to pay for a second thread
      std::copy(in, in + count, out);
      return;
    }
    const std::size_t piece {fault_chunk<T>(count)};
    parallel_for((count + piece - 1) / piece, piece * zero_work, [in, out, count, piece](std::size_t chunk) {
                   const std::size_t begin {chunk * piece};
                   std::copy(in + begin, in + std::min(count, begin + piece), out + begin);
                 });
  }
} // namespace numerical::detail

namespace numerical {

  /*!
   * \brief Caps the threads a kernel uses at \p threads, the caller's included (1: none beside it, and no thread is
   *        started); 0 lifts the cap. For a process that runs several kernels side by side (one per worker process, say), so that
   *        together they do not ask for more threads than the machine has.
   */
  inline void set_max_threads(std::size_t threads) noexcept
  {
    detail::thread_cap().store(threads);
  }

  //! \brief The threads a kernel may use: the cap if one is set, else the hardware threads (starting none).
  [[nodiscard]] inline std::size_t max_threads()
  {
    const std::size_t cap      {detail::thread_cap().load()};
    const std::size_t hardware {detail::hardware_threads()};
    return cap == 0 ? hardware : std::min(cap, hardware);
  }
} // namespace numerical

#endif // NUMERICAL_PARALLEL_HPP
