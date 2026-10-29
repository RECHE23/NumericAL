/*!
 * \file concepts.hpp
 * \brief The scalar types NumericAL's factorisations and solvers take: a field of floating-point numbers, real or
 *        complex. An integer matrix multiplies exactly, but a decomposition of one would divide in integers.
 */
#ifndef NUMERICAL_CONCEPTS_HPP
#define NUMERICAL_CONCEPTS_HPP

#include <concepts>

#include "buffer.hpp"

namespace numerical {

  /*!
   * \brief A floating-point field: `float`, `double`, `long double`, or `std::complex` of one. The factorisations and
   *        the solvers require it, so that an integer matrix is refused at compile time, by name, where it would
   *        otherwise be factorised in integer division.
   */
  template <typename T>
  concept field = std::floating_point<T> || (detail::is_complex_number<T> && std::floating_point<typename T::value_type>);
} // namespace numerical

#endif // NUMERICAL_CONCEPTS_HPP
