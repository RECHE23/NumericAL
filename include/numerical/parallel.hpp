/*!
 * \file parallel.hpp
 * \brief A minimal `std::thread` parallel-for for the hot kernels.
 *
 * The only standard-library parallelism a header-only, zero-dependency leaf can
 * use portably: `std::thread`, no TBB, no execution policies. \ref
 * numerical::detail::parallel_for splits a counted range across the hardware
 * threads — each index is handled by exactly one thread, so a body that writes a
 * disjoint slice of the output per index stays race-free **and deterministic**
 * (the result is identical to the sequential run, bit for bit). Small ranges run
 * sequentially, below a caller-chosen threshold, to avoid thread-spawn overhead.
 */
#ifndef NUMERICAL_PARALLEL_HPP
#define NUMERICAL_PARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace numerical::detail {

  /*!
   * \brief Runs `body(i)` for every `i` in `[0, count)`.
   *
   * When `count >= threshold` the range is partitioned across the hardware
   * threads and run in parallel; otherwise it runs sequentially. \p body must
   * be safe to invoke concurrently for distinct `i` (it is when each call
   * touches a disjoint part of the output).
   *
   * \warning \p body must **not throw**: an exception escaping a worker thread calls
   *          `std::terminate` (undefined behaviour). This is an internal `detail`
   *          helper, used only with the non-throwing numeric kernels, so it does not
   *          pay for per-thread exception capture — the precondition holds by
   *          construction at every call site.
   *
   * \param[in] count     The number of iterations.
   * \param[in] threshold The smallest \p count worth parallelising.
   * \param[in] body       The per-index work (`void(std::size_t)`).
   */
  template <typename Body>
  void parallel_for(std::size_t count,
                    std::size_t threshold,
                    const Body& body)
  {
    if (count < threshold) {
      for (std::size_t i {0}; i < count; ++i) {
        body(i);
      }
      return;
    }
    const std::size_t        threads {std::max<std::size_t>(2, std::thread::hardware_concurrency())};
    const std::size_t        chunk   {((count + threads) - 1) / threads};
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t begin {0}; begin < count; begin += chunk) {
      const std::size_t end {std::min(count, begin + chunk)};
      pool.emplace_back([&body, begin, end] {
                          for (std::size_t i {begin}; i < end; ++i) {
                            body(i);
                          }
                        });
    }
    for (std::thread& worker : pool) {
      worker.join();
    }
  }
} // namespace numerical::detail

#endif // NUMERICAL_PARALLEL_HPP
