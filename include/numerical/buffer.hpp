/*!
 * \file buffer.hpp
 * \brief \ref numerical::buffer, the storage of the library's vectors, matrices and tensors: a std::vector whose
 *        elements constructed with no value are left uninitialised.
 */
#ifndef NUMERICAL_BUFFER_HPP
#define NUMERICAL_BUFFER_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#  include <sys/mman.h>
#endif

namespace numerical {

  namespace detail {

    /*!
     * \brief The allocator of \ref numerical::buffer. An element constructed with no value is default-initialised, so a
     *        `double` keeps whatever its memory held, where std::allocator zeroes it; one constructed with a value is
     *        constructed from it. A result written in full is then written once, by the threads that compute it,
     *        not zeroed first on one of them (a product of order 512 on eight arm64 threads spent a fifth of its time
     *        in that zeroing).
     */
    template <typename T>
    struct default_init_allocator : std::allocator<T>
    {
      using value_type = T; //!< The element type.

      //! \brief The allocator of another element type, as std::allocator_traits rebinds it.
      template <typename U>
      struct rebind
      {
        using other = default_init_allocator<U>; //!< The rebound allocator.
      };

      default_init_allocator() noexcept = default;

      //! \brief From the allocator of another element type: stateless, as std::allocator.
      template <typename U>
      // NOLINTNEXTLINE(hicpp-explicit-conversions,google-explicit-constructor): the containers rebind an allocator by this conversion
      explicit (false) default_init_allocator(const default_init_allocator<U>& /*other*/) noexcept
      {}

      //! \brief Default-initialises the element at \p at.
#if defined(__linux__)
      //! \brief The size of a huge page, and the alignment of a buffer that asks for them.
      static constexpr std::size_t huge_page {std::size_t {1} << 21U};

      //! \brief The bytes from which a buffer asks Linux for huge pages (`MADV_HUGEPAGE`; distributions leave
      //!        transparent huge pages to the program, as `madvise`): fewer translations for the decompositions'
      //!        strided passes (six x86-64 AVX2 cores, order 2048: `inverse` −23 %, `solve` −19 %, `cholesky` −23 %,
      //!        order 3072 −14 to −20 %, order 1448 unchanged). Below it, a fresh result's 2 MiB pages were each faulted
      //!        in, zeroed, on one thread, where the threads that write it share its 4 KiB ones: a product of order 512
      //!        took 1.34 ms, 1.14 without.
      static constexpr std::size_t huge_least {std::size_t {1} << 24U};

      //! \brief The data's offset past its huge page's start goes round 64 multiples of 4 KiB, plus room for the
      //!        start's address: buffers that all began on a 2 MiB boundary fell on the same cache sets (a product of
      //!        order 512 with every operand so aligned: 1.49 ms, 1.15 apart).
      static constexpr std::size_t huge_colours {64};

      //! \brief Whether \p n elements take huge pages: from \ref huge_least bytes, a count whose bytes overflow left to
      //!        std::allocator, which refuses it.
      [[nodiscard]] static constexpr bool on_huge_pages(std::size_t n) noexcept
      {
        return n <= (static_cast<std::size_t>(-1) - huge_page) / sizeof(T) && n * sizeof(T) >= huge_least;
      }

      //! \brief `n` elements: from \ref huge_least bytes on huge pages, the allocation's start kept before the data.
      [[nodiscard]] T* allocate(std::size_t n)
      {
        if (!on_huge_pages(n)) {
          return std::allocator<T>::allocate(n);
        }
        static std::atomic<std::size_t> colour {0};
        const std::size_t               offset {((colour.fetch_add(1) % huge_colours) * 4096) + alignof(std::max_align_t) * 4};
        const std::size_t               bytes  {((offset + (n * sizeof(T))) + huge_page - 1) / huge_page * huge_page};
        auto* const                     start  {static_cast<std::byte*>(::operator new(bytes, std::align_val_t {huge_page}))};
        (void)madvise(start, bytes, MADV_HUGEPAGE); // a hint: the pages stay 4 KiB where the system declines
        std::byte* const data                  {start + offset};
        std::memcpy(data - sizeof(std::byte*), &start, sizeof(std::byte*));
        return reinterpret_cast<T*>(data);          // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): storage for T, aligned past max_align_t
      }

      //! \brief Frees what \ref allocate gave for \p n elements.
      void deallocate(T         * at,
                      std::size_t n) noexcept
      {
        if (!on_huge_pages(n)) {
          std::allocator<T>::deallocate(at, n);
          return;
        }
        std::byte* start {};
        std::memcpy(&start, reinterpret_cast<std::byte*>(at) - sizeof(std::byte*), sizeof(std::byte*)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the bytes before the data
        ::operator delete(start,
                          std::align_val_t {huge_page});
      }

#endif

      template <typename U>
      void construct(U* at) noexcept(std::is_nothrow_default_constructible_v<U>)
      {
        ::new (static_cast<void*>(at)) U;
      }

      //! \brief Constructs the element at \p at from \p args.
      template <typename U,
                typename ... Args>
      void construct(U   *     at,
                     Args&&... args)
      {
        ::new (static_cast<void*>(at)) U(std::forward<Args>(args)...);
      }
    };

    /*!
     * \brief Element-wise equality of a \ref numerical::buffer and a std::vector, found by argument-dependent lookup
     *        through the allocator: code that compared a tensor's elements with a std::vector keeps doing so.
     */
    template <typename T>
    [[nodiscard]] bool operator==(const std::vector<T, default_init_allocator<T>>& buffer,
                                  const std::vector<T>&                            other)
    {
      return std::equal(buffer.begin(), buffer.end(), other.begin(), other.end());
    }
  } // namespace detail

  /*!
   * \brief The storage of \ref vector, \ref matrix and \ref tensor, namely a std::vector whose `buffer<T>(n)` and `resize(n)`
   *        leave a `double` uninitialised, for a result about to be written in full. `buffer<T>(n, T {})` zeroes, as
   *        `std::vector<T>(n)` does.
   */
  template <typename T>
  using buffer = std::vector<T, detail::default_init_allocator<T>>;
} // namespace numerical

#endif // NUMERICAL_BUFFER_HPP
