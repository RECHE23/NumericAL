/*!
 * \file memory.hpp
 * \brief The workspaces NumericAL keeps between calls, given back (\ref numerical::release_memory).
 */
#ifndef NUMERICAL_MEMORY_HPP
#define NUMERICAL_MEMORY_HPP

#include <chrono>

#if defined(__GLIBC__)
#  include <malloc.h>
#elif defined(__APPLE__)
#  include <malloc/malloc.h>
#elif defined(_WIN32)
#  include <malloc.h>
#endif

#include "buffer.hpp"
#include "parallel.hpp"

namespace numerical {

  namespace detail {

    //! \brief \ref release_thread_memory, then the allocator's free pages to the system, on the calling thread: glibc
    //!        keeps an arena for each thread that allocates, whose free pages the thread trims best itself.
    inline void release_and_trim()
    {
      release_thread_memory();
#if defined(__GLIBC__)
      static_cast<void>(malloc_trim(0));
#endif
    }

    //! \brief \ref numerical::release_memory with \p pool's workers, none where it is nullptr (a pool never started
    //!        holds nothing, and starting it would cost threads, not free memory).
    inline void release_memory_on(thread_pool* pool)
    {
      release_thread_memory();
      if (pool != nullptr) {
        static_cast<void>(pool->release_on_workers(&release_and_trim, std::chrono::seconds {1}));
      }
#if defined(__linux__)
      kept_huge_blocks::shared().clear();
#endif
      kept_page_blocks::shared().clear();
      // The allocator's own free pages to the system too: glibc keeps what its arenas freed (a session of order 1500
      // kept 19 MB after the rest went, none after this), macOS's zones likewise.
#if defined(__GLIBC__)
      static_cast<void>(malloc_trim(0));
#elif defined(__APPLE__)
      static_cast<void>(malloc_zone_pressure_relief(nullptr, 0));
#elif defined(_WIN32)
      static_cast<void>(_heapmin());
#endif
    }
  } // namespace detail

  /*!
   * \brief Frees the workspaces NumericAL keeps between calls, as oneMKL's `mkl_free_buffers`: each thread keeps the
   *        buffers its kernels last needed (a matrix product's packing, a solve's copies, an `eigh`'s merges, an FFT's
   *        scratch), so the next call of a size faults no fresh page in (a solve of order 1000 spent 2.5 ms of its 30
   *        on them), and freed blocks of a few megabytes are kept for the next buffer that fits one. A long-lived
   *        process (a server, a notebook) that is done with large problems can give that memory back.
   *
   *        The calling thread's workspaces go at once, and each worker of the thread pool, if it has started, empties
   *        its own between two runs, never inside one (this waits for them, at most a second); then the kept blocks
   *        go. Another thread's
   *        workspaces stay, its own to free by calling this. Last, the allocator is asked to give its free pages back
   *        to the system (`malloc_trim`, `malloc_zone_pressure_relief`, `_heapmin`). The next call that needs a
   *        workspace makes it anew. Under glibc each thread's arena may keep some freed pages yet (an eight-thread
   *        session of order 1500: 210 MB kept, 34 MB after this; with `MALLOC_ARENA_MAX=1`, 3 MB). Nothing must be
   *        running on the calling thread meanwhile, which holds when it calls this between NumericAL's calls.
   */
  inline void release_memory()
  {
    detail::release_memory_on(detail::thread_pool::started());
  }
} // namespace numerical

#endif // NUMERICAL_MEMORY_HPP
