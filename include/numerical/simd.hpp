/*!
 * \file simd.hpp
 * \brief The native vector of `float` and `double` the kernels hold their lanes in (GNU vector extensions,
 *        under GCC and clang; a kernel keeps a scalar form beside it for other compilers).
 */
#ifndef NUMERICAL_SIMD_HPP
#define NUMERICAL_SIMD_HPP

#include <cstddef>

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
} // namespace numerical::detail

#endif // NUMERICAL_SIMD_HPP
