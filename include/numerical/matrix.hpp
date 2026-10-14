/*!
 * \file matrix.hpp
 * \brief A dense, row-major, value-semantic matrix over a scalar type.
 *
 * \ref numerical::matrix is generic over its element type `T` (the scalar), like
 * \ref numerical::vector. It offers element-wise arithmetic, scalar and
 * matrix/vector products, transpose, and tolerance comparison. Operations with
 * incompatible shapes throw \ref numerical::dimension_error. Factorisations and the
 * linear solve over a matrix live in `decomposition.hpp`.
 */
#ifndef NUMERICAL_MATRIX_HPP
#define NUMERICAL_MATRIX_HPP

#include <algorithm>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

#include "gemm.hpp"
#include "parallel.hpp"
#include "vector.hpp"

namespace numerical {

  /*!
   * \brief A dense, row-major matrix of `T` with value semantics.
   * \tparam T The scalar element type.
   */
  template <typename T>
  class matrix
  {
  public:

    //! \brief An empty matrix (0×0).
    matrix() = default;

    /*!
     * \brief From a braced list of rows; every row must have the same length.
     * \param[in] rows The rows, each a braced list of elements.
     * \throws dimension_error If the rows have unequal lengths.
     */
    matrix(std::initializer_list<std::initializer_list<T>> rows)
      : rows_(rows.size())
    {
      std::size_t r {0};
      for (const std::initializer_list<T>& row : rows) {
        if (r == 0) {
          cols_ = row.size();
        }
        else if (row.size() != cols_) {
          throw dimension_error("matrix rows have unequal lengths");
        }
        data_.insert(data_.end(), row.begin(), row.end());
        ++r;
      }
    }

    /*!
     * \brief From dimensions and a flat row-major buffer.
     * \param[in] rows,cols The dimensions.
     * \param[in] elements  The `rows*cols` elements, row-major (moved in).
     * \throws dimension_error If `elements.size()` is not `rows*cols`.
     */
    explicit matrix(std::size_t rows,
                    std::size_t cols,
                    buffer<T>   elements)
      : rows_(rows), cols_(cols), data_(std::move(elements))
    {
      if (data_.size() != detail::checked_mul(rows_, cols_)) {
        throw dimension_error("matrix buffer size does not match dimensions");
      }
    }

    /*!
     * \brief From dimensions and a flat row-major std::vector, copied into the storage (a \ref buffer moves in instead).
     * \param[in] rows,cols The dimensions.
     * \param[in] elements  The `rows*cols` elements, row-major.
     * \throws dimension_error If `elements.size()` is not `rows*cols`.
     */
    explicit matrix(std::size_t           rows,
                    std::size_t           cols,
                    const std::vector<T>& elements)
      : matrix(rows,
               cols,
               buffer<T>(elements.begin(), elements.end()))
    {}

    /*!
     * \brief From dimensions and a braced list of the elements, row-major.
     * \param[in] rows,cols The dimensions.
     * \param[in] elements  The `rows*cols` elements, row-major.
     * \throws dimension_error If the list's length is not `rows*cols`.
     */
    explicit matrix(std::size_t              rows,
                    std::size_t              cols,
                    std::initializer_list<T> elements)
      : matrix(rows,
               cols,
               buffer<T>(elements))
    {}

    //! \brief The number of rows.
    [[nodiscard]] std::size_t rows() const noexcept
    {
      return rows_;
    }

    //! \brief The number of columns.
    [[nodiscard]] std::size_t cols() const noexcept
    {
      return cols_;
    }

    //! \brief The element at (\p row, \p col). \pre In bounds.
    [[nodiscard]] const T& operator()(std::size_t row,
                                      std::size_t col) const
    {
      return data_[(row * cols_) + col];
    }

    //! \brief Exact equality (same shape and equal elements).
    [[nodiscard]] bool operator==(const matrix& other) const
    {
      return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
    }

    //! \brief Const iterator to the first element (row-major).
    [[nodiscard]] typename buffer<T>::const_iterator begin() const noexcept
    {
      return data_.begin();
    }

    //! \brief Const iterator past the last element.
    [[nodiscard]] typename buffer<T>::const_iterator end() const noexcept
    {
      return data_.end();
    }

    //! \brief The elements, row-major, handed over by a matrix about to expire, which is left `0 × 0`: a large result
    //!        moves on without a copy (each copy of a fresh buffer faults its pages in one by one).
    [[nodiscard]] buffer<T> elements() &&
    {
      rows_ = 0;
      cols_ = 0;
      return std::move(data_);
    }

  private:

    std::size_t    rows_ {0}; //!< The number of rows.
    std::size_t    cols_ {0}; //!< The number of columns.
    buffer<T>      data_;     //!< The elements, row-major.
  };

  //! \brief Element-wise sum. \throws dimension_error If the shapes differ.
  template <typename T>
  [[nodiscard]] matrix<T> operator+(const matrix<T>& a,
                                    const matrix<T>& b)
  {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
      throw dimension_error("matrix shapes differ in '+'");
    }
    buffer<T> out(a.rows() * a.cols());
    std::transform(a.begin(), a.end(), b.begin(), out.begin(), std::plus<T> {});
    return matrix<T> {a.rows(), a.cols(), std::move(out)};
  }

  //! \brief Element-wise difference. \throws dimension_error If the shapes differ.
  template <typename T>
  [[nodiscard]] matrix<T> operator-(const matrix<T>& a,
                                    const matrix<T>& b)
  {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
      throw dimension_error("matrix shapes differ in '-'");
    }
    buffer<T> out(a.rows() * a.cols());
    std::transform(a.begin(), a.end(), b.begin(), out.begin(), std::minus<T> {});
    return matrix<T> {a.rows(), a.cols(), std::move(out)};
  }

  //! \brief Scalar times matrix. \param[in] scalar,m Operands. \return scalar·m.
  template <typename T>
  [[nodiscard]] matrix<T> operator*(const T&         scalar,
                                    const matrix<T>& m)
  {
    buffer<T> out(m.rows() * m.cols());
    std::transform(m.begin(), m.end(), out.begin(),
                   [&scalar](const T& element) { return scalar * element; });
    return matrix<T> {m.rows(), m.cols(), std::move(out)};
  }

  //! \brief Matrix times scalar. \param[in] m,scalar Operands. \return m·scalar.
  template <typename T>
  [[nodiscard]] matrix<T> operator*(const matrix<T>& m,
                                    const T&         scalar)
  {
    return scalar * m;
  }

  //! \brief Matrix–vector product: each element its row's dot product with \p v (\ref detail::matrix_times_vector).
  //!        \throws dimension_error If shapes mismatch.
  template <typename T>
  [[nodiscard]] vector<T> operator*(const matrix<T>& a,
                                    const vector<T>& v)
  {
    if (a.cols() != v.size()) {
      throw dimension_error("matrix–vector product: inner dimensions differ");
    }
    buffer<T> out(a.rows());
    if (a.cols() == 0) {
      std::fill(out.begin(), out.end(), T {}); // each element an empty sum
    }
    else {
      detail::matrix_times_vector(&*a.begin(), a.rows(), a.cols(), &v[0], out.data());
    }
    return vector<T> {std::move(out)};
  }

  //! \brief Matrix–matrix product. \throws dimension_error If shapes mismatch.
  template <typename T>
  [[nodiscard]] matrix<T> operator*(const matrix<T>& a,
                                    const matrix<T>& b)
  {
    if (a.cols() != b.rows()) {
      throw dimension_error("matrix–matrix product: inner dimensions differ");
    }
    // detail::gemm adds to `out`, every element in ascending order of the inner index, whatever the blocking and
    // the number of threads (see gemm.hpp): `out` starts at zero, written by the threads that will compute it.
    buffer<T> out(detail::checked_mul(a.rows(), b.cols())); // guard the cross-product alloc
    detail::parallel_zero(out.data(), out.size());
    if (!out.empty() && a.cols() != 0) {
      detail::gemm(a.rows(), b.cols(), a.cols(), &*a.begin(), a.cols(), &*b.begin(), b.cols(), out.data(), b.cols());
    }
    return matrix<T> {a.rows(), b.cols(), std::move(out)};
  }

  //! \brief The transpose. \param[in] m The matrix. \return An `cols×rows` matrix.
  template <typename T>
  [[nodiscard]] matrix<T> transpose(const matrix<T>& m)
  {
    buffer<T> out;
    out.reserve(m.rows() * m.cols());
    for (std::size_t j {0}; j < m.cols(); ++j) {
      for (std::size_t i {0}; i < m.rows(); ++i) {
        out.push_back(m(i, j));
      }
    }
    return matrix<T> {m.cols(), m.rows(), std::move(out)};
  }

  /*!
   * \brief Element-wise closeness within a tolerance (for inexact scalars).
   * \param[in] a,b       Matrices to compare.
   * \param[in] tolerance Maximum allowed magnitude of each element-wise difference.
   * \return `true` if the shapes match and every element pair is equal or differs by at most
   *         \p tolerance; a NaN is close to nothing.
   */
  template <typename T>
  [[nodiscard]] bool approx_equal(const matrix<T>& a,
                                  const matrix<T>& b,
                                  double           tolerance = 1e-9)
  {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
      return false;
    }
    for (std::size_t i {0}; i < a.rows(); ++i) {
      for (std::size_t j {0}; j < a.cols(); ++j) {
        // Equal, or within the tolerance: a NaN is neither, so it is never close, and two equal
        // infinities are equal although their difference is NaN.
        if (!(a(i, j) == b(i, j)) && !(std::abs(a(i, j) - b(i, j)) <= tolerance)) {
          return false;
        }
      }
    }
    return true;
  }
} // namespace numerical

#endif // NUMERICAL_MATRIX_HPP
