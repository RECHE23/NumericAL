/*!
 * \file parallel.hpp
 * \brief A minimal `std::thread` parallel-for for the hot kernels.
 *
 * The only standard-library parallelism a header-only, zero-dependency leaf can
 * use portably: `std::thread`, no TBB, no execution policies. \ref
 * numerical::detail::parallel_for splits a counted range across the hardware
 * threads — each index is handled by exactly one thread, so a body that writes a
 * disjoint slice of the output per index stays race-free **and deterministic**
 * (the result is identical to the sequential run, bit for bit). The caller states the
 * work of one index; a range runs on as many threads as its work pays for, and
 * sequentially when that is fewer than two, since a thread costs about 10 µs to start.
 */
#ifndef NUMERICAL_PARALLEL_HPP
#define NUMERICAL_PARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <thread>
#include <vector>

namespace numerical::detail {

  //! \brief A chunk of a parallel_for, type-erased: runs `body(i)` for `i` in `[begin, end)`.
  using chunk_runner = void (*)(const void* body, std::size_t begin, std::size_t end);

  /*!
   * \brief The part of \ref parallel_for that does not depend on the body: runs \p chunks chunks of
   *        \p chunk indices up to \p count, one thread each, keeps the exception of each, joins them all,
   *        and rethrows the first in index order.
   * \param[in] run    The chunk runner.
   * \param[in] body   The body it is handed.
   * \param[in] count  The number of iterations.
   * \param[in] chunk  Indices per chunk.
   * \param[in] chunks Chunks.
   */
  inline void run_chunks(chunk_runner run,
                         const void*  body,
                         std::size_t  count,
                         std::size_t  chunk,
                         std::size_t  chunks)
  {
    std::vector<std::exception_ptr> failed(chunks);
    // Joined on every way out, a worker the system refuses to start (std::system_error) included: a
    // joinable thread destroyed unjoined would call std::terminate.
    struct joined_pool
    {
      std::vector<std::thread> workers;

      joined_pool()                              = default;
      joined_pool(const joined_pool&)            = delete;
      joined_pool& operator=(const joined_pool&) = delete;
      joined_pool(joined_pool&&)                 = delete;
      joined_pool& operator=(joined_pool&&)      = delete;
      ~joined_pool()
      {
        for (std::thread& worker : workers) {
          worker.join();
        }
      }
    };
    {
      joined_pool pool;
      pool.workers.reserve(chunks);
      for (std::size_t c {0}; c < chunks; ++c) {
        pool.workers.emplace_back([run, body, &failed, c, chunk, count]() noexcept {
                                    try {
                                      run(body, c * chunk, std::min(count, (c + 1) * chunk));
                                    }
                                    catch (...) {
                                      failed[c] = std::current_exception();
                                    }
                                  });
      }
    }
    for (const std::exception_ptr& error : failed) {
      if (error) {
        std::rethrow_exception(error);
      }
    }
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
    // Asked once: the query is a system call, and costs more than a small product.
    static const std::size_t hardware {std::thread::hardware_concurrency()};
    const std::size_t        threads  {threads_for(count, work_per_index, hardware)};
    if (threads < 2) {
      for (std::size_t i {0}; i < count; ++i) {
        body(i);
      }
      return;
    }
    const std::size_t chunk {((count + threads) - 1) / threads};
    run_chunks([](const void* erased, std::size_t begin, std::size_t end) {
                 const Body& work {*static_cast<const Body*>(erased)};
                 for (std::size_t i {begin}; i < end; ++i) {
                   work(i);
                 }
               },
               &body, count, chunk, ((count + chunk) - 1) / chunk);
  }
} // namespace numerical::detail

#endif // NUMERICAL_PARALLEL_HPP
