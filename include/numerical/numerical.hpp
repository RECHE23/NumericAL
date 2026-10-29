/*!
 * \file numerical.hpp
 * \brief Single public header for NumericAL — include this to use the library.
 *
 * NumericAL is the ecosystem's numerical computing library: a pure, header-only,
 * Scalar-generic set of numeric structures and the algorithms over them. It is a
 * leaf (a sibling of REAL) — it depends on nothing but the C++ standard library
 * and knows nothing of expressions or any symbolic layer; it computes with
 * concrete numbers and arrays of them.
 *
 * This header aggregates the public modules.
 */
#ifndef NUMERICAL_NUMERICAL_HPP
#define NUMERICAL_NUMERICAL_HPP

#include "amx.hpp"
#include "banded.hpp"
#include "blas.hpp"
#include "concepts.hpp"
#include "decomposition.hpp"
#include "eigen.hpp"
#include "fft.hpp"
#include "generalized.hpp"
#include "ldl.hpp"
#include "matrix.hpp"
#include "matrix_functions.hpp"
#include "tensor.hpp"
#include "tensor_ops.hpp"
#include "vector.hpp"

/*!
 * \namespace numerical
 * \brief The NumericAL public API.
 */

#endif // NUMERICAL_NUMERICAL_HPP
