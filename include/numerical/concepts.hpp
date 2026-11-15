/*!
 * \file concepts.hpp
 * \brief The scalar types NumericAL's factorisations and solvers take: a field of floating-point numbers, real or
 *        complex. An integer matrix multiplies exactly, but a decomposition of one would divide in integers.
 */
#ifndef NUMERICAL_CONCEPTS_HPP
#define NUMERICAL_CONCEPTS_HPP

#include <complex>
#include <concepts>

#include "buffer.hpp"

namespace numerical {

  /*!
   * \brief A floating-point field: `float`, `double`, `long double`, or `std::complex` of one. The factorisations and
   *        the solvers require it, so that an integer matrix is refused at compile time, by name, where it would
   *        otherwise be factorised in integer division.
   *
   *        What each carries in its own precision: LU, Cholesky and the unpivoted QR compute in `T` itself, a
   *        `long double` to its own precision. The SVD takes any `field` and computes in `double`: its singular values
   *        are `double`, and a `long double` matrix's factors hold a `double`'s precision. The eigendecompositions, the
   *        Schur form, the pivoted QR and LDLᴴ take `double_field` alone; the symmetric and Hermitian eigensolvers and
   *        the generalized forms are of `double` and `std::complex<double>` by their signatures.
   */
  template <typename T>
  concept field = std::floating_point<T> || (detail::is_complex_number<T> && std::floating_point<typename T::value_type>);

  //! \brief `double` or `std::complex<double>`: the scalars of the decompositions written for those alone (the
  //!        eigendecompositions, the Schur form, the pivoted QR, LDLᴴ), which refuse any other at compile time by this
  //!        name.
  template <typename T>
  concept double_field = std::same_as<T, double> || std::same_as<T, std::complex<double>>;
} // namespace numerical

#endif // NUMERICAL_CONCEPTS_HPP
