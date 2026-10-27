/*!
 * \file buffer.hpp
 * \brief \ref numerical::buffer, the storage of the library's vectors, matrices and tensors: a std::vector whose
 *        elements constructed with no value are left uninitialised.
 */
#ifndef NUMERICAL_BUFFER_HPP
#define NUMERICAL_BUFFER_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <mutex>

// A function kept out of its callers, in the spelling each compiler takes.
#if !defined(NUMERICAL_OUT_OF_LINE)
#  if defined(_MSC_VER) && !defined(__clang__)
#    define NUMERICAL_OUT_OF_LINE __declspec(noinline)
#  else
#    define NUMERICAL_OUT_OF_LINE [[gnu::noinline]] // NOLINT(cppcoreguidelines-macro-usage): an attribute, which no function can stand for
#  endif
#endif

#if defined(__linux__)
#  include <sys/mman.h>
#endif

namespace numerical {

  namespace detail {

    /*!
     * \brief Blocks freed and kept for the next buffer that fits one: a fresh block's pages are each faulted in and
     *        zeroed by the system at their first write, on the thread that writes it (glibc hands a large freed block
     *        back to the system unless the next request is of its very size; macOS's allocator likewise), where a kept
     *        block's are already in place. \p Places blocks at most, each under \p Largest bytes, the newest kept: a
     *        block freed while every place is taken frees the one kept longest. The first allocation one fits, with
     *        no more than half of it unused, takes it. Blocks aligned on \p Alignment.
     */
    template <std::size_t Alignment,
              std::size_t Places,
              std::size_t Largest>
    class kept_blocks
    {
    public:

      //! \brief The alignment of a block.
      static constexpr std::size_t alignment {Alignment};

      //! \brief The bytes from which a freed block is freed at once, not kept.
      static constexpr std::size_t largest {Largest};

      //! \brief The process's blocks, never destroyed: a buffer of static storage may be freed after any destructor.
      [[nodiscard]] static kept_blocks& shared()
      {
        static kept_blocks* const blocks {new kept_blocks}; // NOLINT(cppcoreguidelines-owning-memory): lives as long as the process
        return *blocks;
      }

      kept_blocks() = default;

      //! \brief Frees the blocks still kept.
      ~kept_blocks()
      {
        for (const block& b : kept_) {
          if (b.start != nullptr) {
            ::operator delete(b.start, std::align_val_t {alignment});
          }
        }
      }

      kept_blocks(const kept_blocks&)            = delete;
      kept_blocks& operator=(const kept_blocks&) = delete;
      kept_blocks(kept_blocks&&)                 = delete;
      kept_blocks& operator=(kept_blocks&&)      = delete;

      //! \brief A kept block of at least \p bytes and under twice as many, its size in \p got; nullptr if none.
      [[nodiscard]] std::byte* take(std::size_t  bytes,
                                    std::size_t& got) noexcept
      {
        const std::lock_guard<std::mutex> held {lock_};
        block*                            best {nullptr};
        for (block& b : kept_) {
          if (b.start != nullptr && b.bytes >= bytes && b.bytes / 2 < bytes && (best == nullptr || b.bytes < best->bytes)) {
            best = &b;
          }
        }
        if (best == nullptr) {
          return nullptr;
        }
        got = best->bytes;
        return std::exchange(best->start, nullptr);
      }

      /*!
       * \brief Keeps the block at \p start of \p bytes if it is under \ref largest, in place of the block kept
       *        longest when every place is taken (that one freed): the blocks of the sizes in use stay, those of a
       *        size no longer asked for leave. False, the block not kept, if it is too large.
       */
      [[nodiscard]] bool keep(std::byte*  start,
                              std::size_t bytes) noexcept
      {
        if (bytes >= largest) {
          return false;
        }
        std::byte* evicted {nullptr};
        {
          const std::lock_guard<std::mutex> held  {lock_};
          block*                            place {&kept_[0]};
          for (block& b : kept_) {
            if (b.start == nullptr || b.order < place->order) {
              place = &b;
              if (b.start == nullptr) {
                break;
              }
            }
          }
          evicted = place->start;
          *place  = block {start, bytes, ++kept_count_};
        }
        if (evicted != nullptr) {
          ::operator delete(evicted,
                            std::align_val_t {alignment});
        }
        return true;
      }

    private:

      struct block
      {
        std::byte*  start {nullptr}; //!< The block's start, nullptr for a free place.
        std::size_t bytes {0};       //!< Its size.
        std::size_t order {0};       //!< When it was kept, the least the longest.
      };

      std::mutex                lock_;
      std::array<block, Places> kept_       {};
      std::size_t               kept_count_ {0};
    };

#if defined(__linux__)
    //! \brief The huge-page blocks (2 MiB aligned) of \ref default_init_allocator: two, each under 64 MiB.
    using kept_huge_blocks = kept_blocks<std::size_t {1} << 21U, 2, std::size_t {1} << 26U>;

    //! \brief The bytes from which a buffer asks Linux for huge pages (see \ref default_init_allocator).
    inline constexpr std::size_t huge_pages_least {std::size_t {1} << 24U};

    //! \brief The bytes under which a buffer's block is a kept one of 4 KiB pages: \ref huge_pages_least.
    inline constexpr std::size_t kept_pages_largest {huge_pages_least};
#else
    //! \brief The bytes under which a buffer's block is a kept one, no huge pages asked: 32 MiB, four of them at most
    //!        128 MiB kept.
    inline constexpr std::size_t kept_pages_largest {std::size_t {1} << 25U};
#endif

    /*!
     * \brief The blocks of 4 KiB pages of \ref default_init_allocator, from \ref kept_pages_least bytes: four, each under
     *        \ref kept_pages_largest.
     */
    using kept_page_blocks = kept_blocks<std::size_t {4096}, 4, kept_pages_largest>;

    /*!
     * \brief The fewest bytes a buffer takes a kept block of 4 KiB pages for (\ref kept_page_blocks): below, the
     *        allocator's own free lists reuse what is freed.
     */
    inline constexpr std::size_t kept_pages_least {std::size_t {1} << 20U};

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

#if defined(__linux__)
      //! \brief The size of a huge page, and the alignment of a buffer that asks for them.
      static constexpr std::size_t huge_page {kept_huge_blocks::alignment};

      //! \brief The bytes from which a buffer asks Linux for huge pages (`MADV_HUGEPAGE`; distributions leave
      //!        transparent huge pages to the program, as `madvise`): fewer translations for the decompositions'
      //!        strided passes (six x86-64 AVX2 cores, order 2048: `inverse` −23 %, `solve` −19 %, `cholesky` −23 %,
      //!        order 3072 −14 to −20 %, order 1448 unchanged). Below it, a fresh result's 2 MiB pages were each faulted
      //!        in, zeroed, on one thread, where the threads that write it share its 4 KiB ones: a product of order 512
      //!        took 1.34 ms, 1.14 without.
      static constexpr std::size_t huge_least {huge_pages_least};

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

#endif

      //! \brief The data's offset past a kept block's start goes round 16 multiples of 256 bytes past a cache line that
      //!        holds the block's size and the offset: blocks all page-aligned fall on the same cache sets.
      static constexpr std::size_t page_colours {16};

      //! \brief Whether \p n elements take a block of 4 KiB pages kept when freed (\ref kept_page_blocks): from
      //!        \ref kept_pages_least bytes and under \ref kept_pages_largest (from it, on Linux, huge pages).
      [[nodiscard]] static constexpr bool on_kept_pages(std::size_t n) noexcept
      {
        return n < kept_pages_largest / sizeof(T) && n * sizeof(T) >= kept_pages_least;
      }

      //! \brief `n` elements: on huge pages from their least bytes on Linux, in a kept block of 4 KiB pages from
      //!        \ref kept_pages_least, else std::allocator's; a block's size and the data's distance from its start kept
      //!        before the data.
      [[nodiscard]] T* allocate(std::size_t n)
      {
#if defined(__linux__)
        if (on_huge_pages(n)) {
          static std::atomic<std::size_t> colour {0};
          const std::size_t               offset {((colour.fetch_add(1) % huge_colours) * 4096) + alignof(std::max_align_t) * 4};
          std::size_t                     bytes  {((offset + (n * sizeof(T))) + huge_page - 1) / huge_page * huge_page};
          std::byte*                      start  {kept_huge_blocks::shared().take(bytes, bytes)};
          if (start == nullptr) {
            start = static_cast<std::byte*>(::operator new(bytes, std::align_val_t {huge_page}));
            (void)madvise(start, bytes, MADV_HUGEPAGE);                                        // a hint: the pages stay 4 KiB where the system declines
          }
          return placed(start, offset, bytes);
        }
#endif
        if (on_kept_pages(n)) {
          constexpr std::size_t           page   {kept_page_blocks::alignment};
          static std::atomic<std::size_t> colour {0};
          const std::size_t               offset {64 + ((colour.fetch_add(1) % page_colours) * 256)};
          std::size_t                     bytes  {((offset + (n * sizeof(T))) + page - 1) / page * page};
          std::byte*                      start  {kept_page_blocks::shared().take(bytes, bytes)};
          if (start == nullptr) {
            start = static_cast<std::byte*>(::operator new(bytes, std::align_val_t {page}));
          }
          return placed(start, offset, bytes);
        }
        return std::allocator<T>::allocate(n);
      }

      //! \brief Frees what \ref allocate gave for \p n elements.
      void deallocate(T         * at,
                      std::size_t n) noexcept
      {
#if defined(__linux__)
        if (on_huge_pages(n)) {
          deallocate_kept<kept_huge_blocks>(at);
          return;
        }
#endif
        if (on_kept_pages(n)) {
          deallocate_kept<kept_page_blocks>(at);
          return;
        }
        std::allocator<T>::deallocate(at, n);
      }

      //! \brief The data at \p offset into the block at \p start of \p bytes, the offset and the size kept before it.
      [[nodiscard]] static T* placed(std::byte*  start,
                                     std::size_t offset,
                                     std::size_t bytes) noexcept
      {
        std::byte* const data {start + offset};
        std::memcpy(data - sizeof(std::size_t), &offset, sizeof(std::size_t));      // the start, as the data's distance from it
        std::memcpy(data - (2 * sizeof(std::size_t)), &bytes, sizeof(std::size_t)); // the block's size
        return reinterpret_cast<T*>(data);                                          // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): storage for T, aligned past max_align_t
      }

      /*!
       * \brief Frees the block \ref allocate gave from \p at, kept by \p Kept where it takes it. Out of line: inlined
       *        where a small buffer's size is known (a QR's columns of 40 complex numbers), GCC 14 took the read before
       *        \p at for one past the buffer's bounds (-Warray-bounds), though a buffer of that size never comes here.
       */
      template <typename Kept>
      NUMERICAL_OUT_OF_LINE static void deallocate_kept(T* at) noexcept
      {
        auto* const data   {reinterpret_cast<std::byte*>(at)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the storage's bytes
        std::size_t offset {};
        std::size_t bytes  {};
        std::memcpy(&offset, data - sizeof(std::size_t), sizeof(std::size_t));
        std::memcpy(&bytes, data - (2 * sizeof(std::size_t)), sizeof(std::size_t));
        if (!Kept::shared().keep(data - offset, bytes)) {
          ::operator delete(data - offset,
                            std::align_val_t {Kept::alignment});
        }
      }

      //! \brief Default-initialises the element at \p at.
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
