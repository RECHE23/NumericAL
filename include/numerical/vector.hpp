/*!
 * \file vector.hpp
 * \brief A dense, value-semantic vector over a scalar type.
 *
 * \ref numerical::vector is generic over its element type `T` (the scalar): any
 * type closed under the operations used — `int`, `double`, `std::complex<double>`,
 * a rational, … NumericAL is a leaf library: it knows nothing of expressions or any
 * symbolic layer, only concrete numeric data. Operations that need matching
 * shapes throw \ref numerical::dimension_error on a mismatch.
 */
#ifndef NUMERICAL_VECTOR_HPP
#define NUMERICAL_VECTOR_HPP

#include <algorithm>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numerical {

  //! \brief Thrown when an operation's operands have incompatible shapes.
  class dimension_error : public std::runtime_error
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit dimension_error(const std::string& message)
      : std::runtime_error(message)
    {}
  };

  namespace detail {

    //! \brief `a * b`, throwing \ref numerical::dimension_error if the product overflows
    //!        `std::size_t`. Guards dimension/shape products (element counts, allocation
    //!        sizes) against silent wraparound on absurd inputs. The `a != 0` guard also
    //!        keeps the `max / a` division well-defined.
    [[nodiscard]] inline std::size_t checked_mul(std::size_t a,
                                                 std::size_t b)
    {
      if (a != 0 && b > (std::numeric_limits<std::size_t>::max() / a)) {
        throw dimension_error("dimension product overflows std::size_t");
      }
      return a * b;
    }
  } // namespace detail

  /*!
   * \brief A dense vector of `T` with value semantics.
   * \tparam T The scalar element type.
   */
  template <typename T>
  class vector
  {
  public:

    //! \brief An empty vector (size 0).
    vector() = default;

    //! \brief From a braced list of elements. \param[in] elements The elements.
    vector(std::initializer_list<T> elements)
      : data_(elements)
    {}

    //! \brief From an existing buffer. \param[in] elements The elements (moved in).
    explicit vector(std::vector<T> elements)
      : data_(std::move(elements))
    {}

    //! \brief The number of elements.
    [[nodiscard]] std::size_t size() const noexcept
    {
      return data_.size();
    }

    //! \brief The element at \p index. \pre \p index < \ref size.
    [[nodiscard]] const T& operator[](std::size_t index) const
    {
      return data_[index];
    }

    //! \brief Const iterator to the first element.
    [[nodiscard]] typename std::vector<T>::const_iterator begin() const noexcept
    {
      return data_.begin();
    }

    //! \brief Const iterator past the last element.
    [[nodiscard]] typename std::vector<T>::const_iterator end() const noexcept
    {
      return data_.end();
    }

    //! \brief Exact element-wise equality (same size, equal elements).
    [[nodiscard]] bool operator==(const vector& other) const
    {
      return data_ == other.data_;
    }

  private:

    std::vector<T> data_; //!< The elements.
  };

  //! \brief Element-wise sum. \throws dimension_error If the sizes differ.
  template <typename T>
  [[nodiscard]] vector<T> operator+(const vector<T>& a,
                                    const vector<T>& b)
  {
    if (a.size() != b.size()) {
      throw dimension_error("vector sizes differ in '+'");
    }
    std::vector<T> out(a.size());
    std::transform(a.begin(), a.end(), b.begin(), out.begin(), std::plus<T> {});
    return vector<T> {std::move(out)};
  }

  //! \brief Element-wise difference. \throws dimension_error If the sizes differ.
  template <typename T>
  [[nodiscard]] vector<T> operator-(const vector<T>& a,
                                    const vector<T>& b)
  {
    if (a.size() != b.size()) {
      throw dimension_error("vector sizes differ in '-'");
    }
    std::vector<T> out(a.size());
    std::transform(a.begin(), a.end(), b.begin(), out.begin(), std::minus<T> {});
    return vector<T> {std::move(out)};
  }

  //! \brief Scalar times vector. \param[in] scalar,v Operands. \return scalar·v.
  template <typename T>
  [[nodiscard]] vector<T> operator*(const T&         scalar,
                                    const vector<T>& v)
  {
    std::vector<T> out(v.size());
    std::transform(v.begin(), v.end(), out.begin(),
                   [&scalar](const T& element) { return scalar * element; });
    return vector<T> {std::move(out)};
  }

  //! \brief Vector times scalar. \param[in] v,scalar Operands. \return v·scalar.
  template <typename T>
  [[nodiscard]] vector<T> operator*(const vector<T>& v,
                                    const T&         scalar)
  {
    return scalar * v;
  }

  //! \brief Dot product. \throws dimension_error If the sizes differ.
  template <typename T>
  [[nodiscard]] T dot(const vector<T>& a,
                      const vector<T>& b)
  {
    if (a.size() != b.size()) {
      throw dimension_error("vector sizes differ in 'dot'");
    }
    // std::inner_product is sequential (left-to-right), so the reduction order —
    // and thus the floating-point result — stays deterministic.
    return std::inner_product(a.begin(), a.end(), b.begin(), T {});
  }

  /*!
   * \brief Element-wise closeness within a tolerance (for inexact scalars).
   * \param[in] a,b       Vectors to compare.
   * \param[in] tolerance Maximum allowed magnitude of each element-wise difference.
   * \return `true` if the sizes match and every element pair is equal or differs by at most
   *         \p tolerance; a NaN is close to nothing.
   */
  template <typename T>
  [[nodiscard]] bool approx_equal(const vector<T>& a,
                                  const vector<T>& b,
                                  double           tolerance = 1e-9)
  {
    if (a.size() != b.size()) {
      return false;
    }
    for (std::size_t i {0}; i < a.size(); ++i) {
      // Equal, or within the tolerance: a NaN is neither, so it is never close, and two equal
      // infinities are equal although their difference is NaN.
      if (!(a[i] == b[i]) && !(std::abs(a[i] - b[i]) <= tolerance)) {
        return false;
      }
    }
    return true;
  }
} // namespace numerical

#endif // NUMERICAL_VECTOR_HPP
