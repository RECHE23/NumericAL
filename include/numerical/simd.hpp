/*!
 * \file simd.hpp
 * \brief The native vector of `float` and `double` the kernels hold their lanes in (GNU vector extensions,
 *        under GCC and clang; a kernel keeps a scalar form beside it for other compilers).
 */
#ifndef NUMERICAL_SIMD_HPP
#define NUMERICAL_SIMD_HPP

#include <array>
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

  //! \brief Whether `T` is held in \ref native_vector.
  template <typename T>
  concept native_scalar = std::is_same_v<T, float> || std::is_same_v<T, double>;

  /*!
   * \brief `Σ x[i] · y[i]` over native vectors: four of them accumulate independently, lane by lane, then
   *        their lanes and the remainder are summed in a fixed order. A single running sum waits on each
   *        addition before the next; this sum is as deterministic, in another order.
   */
  template <native_scalar T>
  [[nodiscard]] T dot_product(const T*    x,
                              const T*    y,
                              std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t   width {native_vector_bytes / sizeof(T)};
    constexpr std::size_t   lanes {4};
    std::array<lane, lanes> sum   {};
    const std::size_t       done  {count - (count % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
#  pragma GCC unroll 4 // the sums in registers: GCC keeps a rolled array of vectors in memory
      for (std::size_t l {0}; l < lanes; ++l) {
        lane xv;
        lane yv;
        std::memcpy(&xv, x + i + (l * width), sizeof(lane));
        std::memcpy(&yv, y + i + (l * width), sizeof(lane));
        sum[l] += xv * yv;
      }
    }
    T total {};
    for (std::size_t l {0}; l < lanes; ++l) {
      for (std::size_t k {0}; k < width; ++k) {
        total += sum[l][k];
      }
    }
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    return total;
  }

#endif

#if defined(__GNUC__) && !defined(__clang__)

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

  //! \brief `Σ x[i] · y[i]`, in order, for a type no native vector holds.
  template <typename T>
  [[nodiscard]] T dot_product(const T*    x,
                              const T*    y,
                              std::size_t count)
  {
    T total {};
    for (std::size_t i {0}; i < count; ++i) {
      total += x[i] * y[i];
    }
    return total;
  }

  /*!
   * \brief `y[i] -= f · x[i]` for \p count elements of \p y and \p x, which do not overlap. Each element is
   *        computed exactly as that scalar statement computes it (its product rounded before its difference, or
   *        fused with it where the compiler contracts the statement); the arguments follow the expression, since
   *        a compiler may contract `f · x` and `x · f` of complex numbers differently. Under GCC, over native
   *        vectors where `T` has them: at `-O2` GCC leaves the scalar loop scalar, unable to rule out an overlap
   *        without a check it will not add. Clang vectorises the scalar loop itself, with that check, and better
   *        than these vectors (an LU solve of order 128 on arm64 took 19 % longer through them).
   */
  template <typename T>
  void subtract_product(T*          y,
                        const T&    f,
                        const T*    x,
                        std::size_t count)
  {
    const T factor {f}; // read once, as a scalar loop would: it may be an element of y's array
    for (std::size_t i {subtract_scaled_vectors(y, x, factor, count)}; i < count; ++i) {
      y[i] -= factor * x[i];
    }
  }

  //! \brief `y[i] -= x[i] · f`, as the other \ref subtract_product (a real product is the same either way round).
  template <typename T>
  void subtract_product(T*          y,
                        const T*    x,
                        const T&    f,
                        std::size_t count)
  {
    const T factor {f}; // read once, as a scalar loop would: it may be an element of y's array
    for (std::size_t i {subtract_scaled_vectors(y, x, factor, count)}; i < count; ++i) {
      y[i] -= x[i] * factor;
    }
  }

  //! \brief `y[i] += f · x[i]`, as \ref subtract_product.
  template <typename T>
  void add_product(T*          y,
                   const T&    f,
                   const T*    x,
                   std::size_t count)
  {
    const T factor {f}; // read once, as a scalar loop would: it may be an element of y's array
    for (std::size_t i {add_scaled_vectors(y, x, factor, count)}; i < count; ++i) {
      y[i] += factor * x[i];
    }
  }
} // namespace numerical::detail

#endif // NUMERICAL_SIMD_HPP
