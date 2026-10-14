/*!
 * \file buffer.hpp
 * \brief \ref numerical::buffer, the storage of the library's vectors, matrices and tensors: a std::vector whose
 *        elements constructed with no value are left uninitialised.
 */
#ifndef NUMERICAL_BUFFER_HPP
#define NUMERICAL_BUFFER_HPP

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

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
