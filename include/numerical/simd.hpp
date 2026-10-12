/*!
 * \file simd.hpp
 * \brief The native vector of `float` and `double` the kernels hold their lanes in (GNU vector extensions,
 *        under GCC and clang; a kernel keeps a scalar form beside it for other compilers).
 */
#ifndef NUMERICAL_SIMD_HPP
#define NUMERICAL_SIMD_HPP

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace numerical::detail {

#if defined(__GNUC__)
  //! \brief The native vector width, in bytes.
#  if defined(__AVX__)
  inline constexpr std::size_t native_vector_bytes {32};
#  else
  inline constexpr std::size_t native_vector_bytes {16};
#  endif

  //! \brief A native vector of `T`, for the types the kernels hold in vectors.
  template <typename T>
  struct native_vector;

  //! \brief A native vector of `float`.
  template <>
  struct native_vector<float>
  {
    using type = float __attribute__((vector_size(native_vector_bytes))); //!< The vector.
  };

  //! \brief A native vector of `double`.
  template <>
  struct native_vector<double>
  {
    using type = double __attribute__((vector_size(native_vector_bytes))); //!< The vector.
  };

#endif

#if defined(__GNUC__) && !defined(__clang__)
  //! \brief Whether `T` is held in \ref native_vector.
  template <typename T>
  concept native_scalar = std::is_same_v<T, float> || std::is_same_v<T, double>;

  //! \brief The leading elements of `y[i] -= f · x[i]`, a whole number of native vectors: how many it did.
  template <native_scalar T>
  std::size_t subtract_scaled_vectors(T*          y,
                                      const T*    x,
                                      const T&    f,
                                      std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    const std::size_t     done  {count - (count % width)};
    for (std::size_t i {0}; i < done; i += width) {
      lane yv;
      lane xv;
      std::memcpy(&yv, y + i, sizeof(lane));
      std::memcpy(&xv, x + i, sizeof(lane));
      yv -= f * xv;
      std::memcpy(y + i, &yv, sizeof(lane));
    }
    return done;
  }

  //! \brief The leading elements of `y[i] += f · x[i]`, a whole number of native vectors: how many it did.
  template <native_scalar T>
  std::size_t add_scaled_vectors(T*          y,
                                 const T*    x,
                                 const T&    f,
                                 std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    const std::size_t     done  {count - (count % width)};
    for (std::size_t i {0}; i < done; i += width) {
      lane yv;
      lane xv;
      std::memcpy(&yv, y + i, sizeof(lane));
      std::memcpy(&xv, x + i, sizeof(lane));
      yv += f * xv;
      std::memcpy(y + i, &yv, sizeof(lane));
    }
    return done;
  }

#endif

  //! \brief No leading elements in vectors, for a type no native vector holds.
  template <typename T>
  std::size_t subtract_scaled_vectors(T* /*y*/,
                                      const T* /*x*/,
                                      const T& /*f*/,
                                      std::size_t /*count*/)
  {
    return 0;
  }

  //! \brief No leading elements in vectors, for a type no native vector holds.
  template <typename T>
  std::size_t add_scaled_vectors(T* /*y*/,
                                 const T* /*x*/,
                                 const T& /*f*/,
                                 std::size_t /*count*/)
  {
    return 0;
  }

  /*!
   * \brief `y[i] -= f · x[i]` for \p count elements of \p y and \p x, which do not overlap. Each element is
   *        computed exactly as the scalar statement computes it (its product rounded before its difference, or
   *        fused with it where the compiler contracts that statement). Under GCC, over native vectors where `T` has
   *        them: at `-O2` GCC leaves the scalar loop scalar, unable to rule out an overlap without a check it will
   *        not add. Clang vectorises the scalar loop itself, with that check, and better than these vectors
   *        (an LU solve of order 128 on arm64 took 19 % longer through them).
   */
  template <typename T>
  void subtract_scaled(T*          y,
                       const T*    x,
                       const T&    f,
                       std::size_t count)
  {
    for (std::size_t i {subtract_scaled_vectors(y, x, f, count)}; i < count; ++i) {
      y[i] -= f * x[i];
    }
  }

  //! \brief `y[i] += f · x[i]` for \p count elements of \p y and \p x, which do not overlap, as \ref subtract_scaled.
  template <typename T>
  void add_scaled(T*          y,
                  const T*    x,
                  const T&    f,
                  std::size_t count)
  {
    for (std::size_t i {add_scaled_vectors(y, x, f, count)}; i < count; ++i) {
      y[i] += f * x[i];
    }
  }
} // namespace numerical::detail

#endif // NUMERICAL_SIMD_HPP
