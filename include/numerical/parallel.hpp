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
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
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

  //! \brief The pool's threads from the \p logical hardware threads and the \p physical cores (0: unknown): the
  //!        cores where known, at least 1.
  [[nodiscard]] constexpr std::size_t pool_threads(std::size_t logical,
                                                   std::size_t physical) noexcept
  {
    const std::size_t hardware {std::max<std::size_t>(1, logical)};
    return physical == 0 ? hardware : std::min(physical, hardware);
  }

  /*!
   * \brief The threads the pool runs, the caller's included (at least 1), known without starting the pool: one per
   *        physical core where the system tells them (else one per hardware thread). Two threads on one core's
   *        hyperthreads share its multiply-add units: a product of order 128 on eight AVX2 cores with sixteen
   *        hyperthreads took 140 µs on sixteen threads, 71 on eight; order 2048, the same either way. Asked once,
   *        as the pool's workers are started once (glibc reads a file for the hardware threads on every call).
   */
  [[nodiscard]] inline std::size_t hardware_threads() noexcept
  {
    static const std::size_t threads {pool_threads(std::thread::hardware_concurrency(), physical_cores())};
    return threads;
  }

  //! \brief How long a thread polls before it sleeps: kernels run in quick succession (a matrix product packs, then
  //!        computes, panel after panel), and a thread woken from sleep starts tens of microseconds late.
  inline constexpr std::chrono::microseconds spin_time {100};

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
      wake_.notify_all();
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
        std::thread {[this, cores, core] {
                       pin_to_core(cores, core);
                       serve(core);
                     }}.detach();
        ++workers_;
      }
#else
      for (const std::size_t core : worker_cores(0, hardware_threads())) {
        std::thread {[this, core] { serve(core); }}.detach();
        ++workers_;
      }
#endif
    }

    //! \brief A worker on \p core: waits for a run (polling a while, then asleep), works on it unless it sits it
    //!        out, waits for the next.
    [[noreturn]] void serve([[maybe_unused]] std::size_t core)
    {
      std::size_t seen {0};
#if defined(__linux__)
      bool spin        {true}; // the worker that sat a run out sleeps till the next: polling, it slowed its core's caller
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
        std::shared_ptr<pool_job> job;
        {
          std::unique_lock<std::mutex> held {lock_};
          wake_.wait(held, started);
          seen = generation_.load();
          job  = current_;
        }
#if defined(__linux__) // elsewhere no worker is pinned, so none sits out
        spin = !sits_out(core, job->core);
        if (!spin) {
          continue;
        }
#endif
        if (job->joined.fetch_add(1) < job->helpers) { // as many workers as the run's threads, the caller aside
          job->work();
        }
      }
    }

    std::mutex                lock_;                           //!< Guards current_ and generation_.
    std::condition_variable   wake_;                           //!< Signalled when a run starts.
    std::shared_ptr<pool_job> current_;                        //!< The latest run.
    std::atomic<std::size_t>  generation_ {0};                 //!< Counts the runs (polled without the lock).
    std::atomic<bool>         busy_       {false};             //!< Whether a run holds the pool.
    std::size_t               workers_    {0};                 //!< The threads started.
#if defined(__linux__)
    std::vector<std::size_t> processor_cores_;                 //!< Each processor's core (`processor_cores`), set before the workers start.
#endif
    long                      owner_      {current_process()}; //!< The process whose threads they are.
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
    const std::size_t cap     {thread_cap().load()};
    const std::size_t threads {threads_for(count, work_per_index, cap == 0 ? hardware_threads() : std::min(cap, hardware_threads()))};
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
