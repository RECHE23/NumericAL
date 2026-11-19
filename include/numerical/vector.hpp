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
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "buffer.hpp"

namespace numerical {

  /*!
   * \brief What every exception NumericAL throws is, beside its standard base (all but `std::bad_alloc` and
   *        `std::system_error`, the allocator's and the system's): `catch (const numerical::error&)` catches each of them, and `catch (const std::domain_error&)` (its standard base) still catches a \ref
   *        non_finite_error. Not a `std::exception` itself, so that each keeps the standard base it has; \ref what
   *        gives the message.
   */
  class error
  {
  public:

    virtual ~error() = default;

    //! \brief The message, as `std::exception::what` gives it.
    [[nodiscard]] virtual const char* what() const noexcept = 0;

  protected:

    error()                        = default;
    error(const error&)            = default;
    error(error&&)                 = default;
    error& operator=(const error&) = default;
    error& operator=(error&&)      = default;
  };

  namespace detail {
    //! \brief A standard exception `Std` that \ref numerical::error tags too, its one `what` answering for both bases.
    template <typename Std>
    class tagged : public Std, public error
    {
    public:

      using Std::Std;

      //! \brief The message.
      [[nodiscard]] const char* what() const noexcept override
      {
        return Std::what();
      }
    };
  } // namespace detail

  //! \brief Thrown when an operation's operands have incompatible shapes.
  class dimension_error : public detail::tagged<std::runtime_error>
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit dimension_error(const std::string& message)
      : tagged(message)
    {}
  };

  //! \brief Thrown when an argument is outside what an operation takes (a tolerance below 0, an FFT type past 4).
  class argument_error : public detail::tagged<std::invalid_argument>
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit argument_error(const std::string& message)
      : tagged(message)
    {}
  };

  //! \brief Thrown when an index or a range leaves its axis.
  class index_error : public detail::tagged<std::out_of_range>
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit index_error(const std::string& message)
      : tagged(message)
    {}
  };

  namespace detail {

    /*!
     * \brief Whether \p count indices from \p first, \p step apart, all lie in `[0, extent)`: none is out for a count
     *        of 0. In unsigned arithmetic, the last index's distance from \p first never formed where it would pass
     *        the axis, so that no step and no count overflows (`(count - 1) · step` in `std::ptrdiff_t` did, a step
     *        of `PTRDIFF_MIN` and a count of 3 among them, and could then pass the test).
     */
    [[nodiscard]] constexpr bool range_within(std::size_t    first,
                                              std::size_t    count,
                                              std::ptrdiff_t step,
                                              std::size_t    extent) noexcept
    {
      if (count == 0) {
        return true;
      }
      if (first >= extent) {
        return false;
      }
      const std::size_t room      {step >= 0 ? extent - 1 - first : first}; // the indices past first on the step's side
      const std::size_t magnitude {step >= 0 ? static_cast<std::size_t>(step) : std::size_t {0} - static_cast<std::size_t>(step)};
      return magnitude == 0 || count - 1 <= room / magnitude;
    }

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

    /*!
     * \brief The elements a shape's extents `[first, last)` hold: 0 where one of them is 0, whatever the others,
     *        else their product (\ref checked_mul). The extents a tensor keeps, past one it holds 0 of, may multiply
     *        past `std::size_t`: a result of them is empty, not of a wrapped size.
     * \throws dimension_error If none is 0 and their product overflows.
     */
    template <typename It>
    [[nodiscard]] std::size_t element_count(It first,
                                            It last)
    {
      if (std::find(first, last, std::size_t {0}) != last) {
        return 0;
      }
      std::size_t count {1};
      for (; first != last; ++first) {
        count = checked_mul(count, *first);
      }
      return count;
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
    explicit vector(buffer<T> elements)
      : data_(std::move(elements))
    {}

    //! \brief From a std::vector, copied into the storage (a \ref buffer moves in instead). \param[in] elements The
    //!        elements.
    explicit vector(const std::vector<T>& elements)
      : data_(elements.begin(), elements.end())
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
    [[nodiscard]] typename buffer<T>::const_iterator begin() const noexcept
    {
      return data_.begin();
    }

    //! \brief Const iterator past the last element.
    [[nodiscard]] typename buffer<T>::const_iterator end() const noexcept
    {
      return data_.end();
    }

    //! \brief Exact element-wise equality (same size, equal elements).
    [[nodiscard]] bool operator==(const vector& other) const
    {
      return data_ == other.data_;
    }

  private:

    buffer<T> data_; //!< The elements.
  };

  //! \brief Element-wise sum. \throws dimension_error If the sizes differ.
  template <typename T>
  [[nodiscard]] vector<T> operator+(const vector<T>& a,
                                    const vector<T>& b)
  {
    if (a.size() != b.size()) {
      throw dimension_error("vector sizes differ in '+'");
    }
    buffer<T> out(a.size());
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
    buffer<T> out(a.size());
    std::transform(a.begin(), a.end(), b.begin(), out.begin(), std::minus<T> {});
    return vector<T> {std::move(out)};
  }

  //! \brief Scalar times vector. \param[in] scalar,v Operands. \return scalar·v.
  template <typename T>
  [[nodiscard]] vector<T> operator*(const T&         scalar,
                                    const vector<T>& v)
  {
    buffer<T> out(v.size());
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
