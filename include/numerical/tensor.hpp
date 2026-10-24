/*!
 * \file tensor.hpp
 * \brief A dense, row-major, N-dimensional tensor and Einstein summation.
 *
 * \ref numerical::tensor is generic over its element type `T` (the scalar) and over
 * its rank: rank-1 is a vector, rank-2 a matrix, rank-0 a scalar. It is the
 * unifying primitive of the library's vector/tensor algebra. A tensor is built
 * directly from a shape and a row-major buffer, or from a \ref numerical::vector or
 * \ref numerical::matrix (its rank-1 / rank-2 special cases).
 *
 * \ref numerical::einsum evaluates an Einstein-summation specification such as
 * `"ij,jk->ik"` (matrix product), `"i,i->"` (dot product), `"ij->ji"`
 * (transpose), `"ii->"` (trace) or `"i,j->ij"` (outer product), over one or more
 * operands. Each single letter is an index label; a label shared by several axes
 * ties them together; a label absent from the output is summed over. Labels are
 * ASCII letters. Summation runs in a deterministic order (labels sorted
 * ascending), so results are reproducible; a plain dot product (`"i,i->"`) sums in a fixed vectorised order
 * instead, the same on any number of threads (`detail::chunked_dot`).
 */
#ifndef NUMERICAL_TENSOR_HPP
#define NUMERICAL_TENSOR_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gemm.hpp"
#include "matrix.hpp"
#include "parallel.hpp"
#include "vector.hpp"

namespace numerical {

  //! \brief Thrown when an einsum specification is malformed or inconsistent.
  class einsum_error : public std::runtime_error
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit einsum_error(const std::string& message)
      : std::runtime_error(message)
    {}
  };

  /*!
   * \brief A dense, row-major, N-dimensional tensor of `T` with value semantics.
   * \tparam T The scalar element type.
   */
  template <typename T>
  class tensor
  {
  public:

    /*!
     * \brief From a shape and a flat row-major buffer.
     * \param[in] shape    The size of each axis (empty ⇒ a rank-0 scalar).
     * \param[in] elements The `prod(shape)` elements, row-major (moved in).
     * \throws dimension_error If `elements.size()` is not `prod(shape)`.
     */
    tensor(std::vector<std::size_t> shape,
           buffer<T>                elements)
      : shape_(std::move(shape)), data_(std::move(elements))
    {
      finalize();
    }

    /*!
     * \brief From a shape and a flat row-major std::vector, copied into the storage (a \ref buffer moves in instead).
     * \param[in] shape    The size of each axis.
     * \param[in] elements The `prod(shape)` elements, row-major.
     * \throws dimension_error If `elements.size()` is not `prod(shape)`.
     */
    tensor(std::vector<std::size_t> shape,
           const std::vector<T>&    elements)
      : tensor(std::move(shape),
               buffer<T>(elements.begin(), elements.end()))
    {}

    //! \brief From a shape and a braced list of the elements, row-major.
    tensor(std::vector<std::size_t> shape,
           std::initializer_list<T> elements)
      : tensor(std::move(shape),
               buffer<T>(elements))
    {}

    //! \brief A rank-1 tensor from a vector. \param[in] v The vector.
    explicit tensor(const vector<T>& v)
      : shape_ {v.size()}
    {
      data_.reserve(v.size());
      for (std::size_t i {0}; i < v.size(); ++i) {
        data_.push_back(v[i]);
      }
      finalize();
    }

    //! \brief A rank-2 tensor taking over an expiring matrix's elements, without a copy. \param[in] m The matrix.
    explicit tensor(matrix<T>&& m)
      : shape_ {m.rows(), m.cols()}, data_ {std::move(m).elements()}
    {
      finalize();
    }

    //! \brief A rank-2 tensor from a matrix. \param[in] m The matrix.
    explicit tensor(const matrix<T>& m)
      : shape_ {m.rows(), m.cols()}, data_(m.begin(), m.end())
    {
      finalize();
    }

    //! \brief The number of axes (0 for a scalar).
    [[nodiscard]] std::size_t rank() const noexcept
    {
      return shape_.size();
    }

    //! \brief The total number of elements.
    [[nodiscard]] std::size_t size() const noexcept
    {
      return data_.size();
    }

    //! \brief The size of each axis.
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept
    {
      return shape_;
    }

    //! \brief The elements, row-major.
    [[nodiscard]] const buffer<T>& data() const noexcept
    {
      return data_;
    }

    //! \brief The element at a multi-index. \pre Its length is \ref rank, in bounds.
    [[nodiscard]] const T& operator()(const std::vector<std::size_t>& index) const
    {
      std::size_t flat {0};
      for (std::size_t axis {0}; axis < index.size(); ++axis) {
        flat += strides_[axis] * index[axis];
      }
      return data_[flat];
    }

    //! \brief Exact equality (same shape and equal elements).
    [[nodiscard]] bool operator==(const tensor& other) const
    {
      return shape_ == other.shape_ && data_ == other.data_;
    }

  private:

    //! \brief Computes row-major strides and checks the buffer against the shape.
    void finalize()
    {
      strides_.assign(shape_.size(), 1);
      std::size_t acc {1};
      for (std::size_t axis {shape_.size()}; axis-- > 0;) {
        strides_[axis] = acc;
        acc            = detail::checked_mul(acc, shape_[axis]); // guard the element-count product
      }
      if (data_.size() != acc) {
        throw dimension_error("tensor: data size does not match the shape");
      }
    }

    std::vector<std::size_t> shape_;   //!< The size of each axis.
    std::vector<std::size_t> strides_; //!< Row-major strides.
    buffer<T>                data_;    //!< The elements, row-major.
  };

  /*!
   * \brief Element-wise closeness within a tolerance (for inexact scalars).
   * \param[in] a,b       Tensors to compare.
   * \param[in] tolerance Maximum allowed magnitude of each element-wise difference.
   * \return `true` if the shapes match and every element pair is equal or differs by at most
   *         \p tolerance; a NaN is close to nothing.
   */
  template <typename T>
  [[nodiscard]] bool approx_equal(const tensor<T>& a,
                                  const tensor<T>& b,
                                  double           tolerance = 1e-9)
  {
    if (a.shape() != b.shape()) {
      return false;
    }
    for (std::size_t i {0}; i < a.size(); ++i) {
      // Equal, or within the tolerance: a NaN is neither, so it is never close, and two equal
      // infinities are equal although their difference is NaN.
      if (!(a.data()[i] == b.data()[i]) && !(std::abs(a.data()[i] - b.data()[i]) <= tolerance)) {
        return false;
      }
    }
    return true;
  }

  namespace detail {
    //! \brief A shape as NumPy prints it, `(2, 3)`, for a message.
    [[nodiscard]] inline std::string shape_text(const std::vector<std::size_t>& shape)
    {
      std::string out {"("};
      for (std::size_t i {0}; i < shape.size(); ++i) {
        out += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
      }
      return out + (shape.size() == 1 ? ",)" : ")");
    }

    //! \brief The row-major strides of \p shape, with a stride of 0 on each axis of extent 1 (it broadcasts),
    //!        aligned on the last axis of a result of rank \p rank (missing leading axes stride 0 too).
    [[nodiscard]] inline std::vector<std::size_t> broadcast_strides(const std::vector<std::size_t>& shape,
                                                                    std::size_t                     rank)
    {
      std::vector<std::size_t> strides(rank, 0);
      std::size_t              stride {1};
      for (std::size_t i {shape.size()}; i-- > 0;) {
        strides[rank - shape.size() + i] = shape[i] == 1 ? 0 : stride;
        stride                          *= shape[i];
      }
      return strides;
    }
  } // namespace detail

  /*!
   * \brief The shape two shapes broadcast to, as NumPy's `broadcast_shapes`.
   * \throws dimension_error If an axis has two extents, neither of them 1.
   */
  [[nodiscard]] inline std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t>& a,
                                                                const std::vector<std::size_t>& b)
  {
    const std::size_t        rank {std::max(a.size(), b.size())};
    std::vector<std::size_t> out(rank);
    for (std::size_t i {0}; i < rank; ++i) {
      const std::size_t x {i < rank - a.size() ? 1 : a[i - (rank - a.size())]};
      const std::size_t y {i < rank - b.size() ? 1 : b[i - (rank - b.size())]};
      if (x != y && x != 1 && y != 1) {
        throw dimension_error("operands could not be broadcast together with shapes " + detail::shape_text(a) + " "
                              + detail::shape_text(b));
      }
      out[i] = x == 1 ? y : x;
    }
    return out;
  }

  namespace detail {
    /*!
     * \brief `op(a, b)` element by element of the row-major \p a (\p a_shape) and \p b (\p b_shape), broadcast against
     *        each other (\ref broadcast_apply's, its operands read where they lie).
     * \throws dimension_error If the shapes do not broadcast.
     */
    template <typename T,
              typename Op>
    [[nodiscard]] tensor<T> broadcast_into(const T*                        a,
                                           const std::vector<std::size_t>& a_shape,
                                           const T*                        b,
                                           const std::vector<std::size_t>& b_shape,
                                           const Op&                       op)
    {
      if (a_shape == b_shape) {
        const std::size_t size   {detail::element_count(a_shape.begin(), a_shape.end())};
        buffer<T>         out(size);
        const std::size_t chunks {(size + detail::elementwise_chunk - 1) / detail::elementwise_chunk};
        detail::parallel_for(chunks, detail::elementwise_chunk, [&](std::size_t c) {
                               const std::size_t first {c * detail::elementwise_chunk};
                               const std::size_t last {std::min(first + detail::elementwise_chunk, size)};
                               std::transform(a + first, a + last, b + first, out.data() + first, op);
                             });
        return tensor<T> {a_shape, std::move(out)};
      }
      std::vector<std::size_t>       shape     {broadcast_shape(a_shape, b_shape)};
      const std::size_t              full_rank {shape.size()};
      const std::vector<std::size_t> full_sa   {detail::broadcast_strides(a_shape, full_rank)};
      const std::vector<std::size_t> full_sb   {detail::broadcast_strides(b_shape, full_rank)};
      const std::size_t              count     {detail::element_count(shape.begin(), shape.end())};
      buffer<T>                      out(count); // every element written below
      if (count == 0) {
        return tensor<T> {std::move(shape), std::move(out)};
      }
      // Neighbouring axes each operand steps through as one (its stride on the outer the inner's times the inner's
      // extent, a broadcast axis 0 on both) run as one axis: the same elements in the same order.
      std::vector<std::size_t> dims {shape[0]};
      std::vector<std::size_t> sa   {full_sa[0]};
      std::vector<std::size_t> sb   {full_sb[0]};
      for (std::size_t axis {1}; axis < full_rank; ++axis) {
        if (sa.back() == full_sa[axis] * shape[axis] && sb.back() == full_sb[axis] * shape[axis]) {
          dims.back() *= shape[axis];
          sa.back()    = full_sa[axis];
          sb.back()    = full_sb[axis];
          continue;
        }
        dims.push_back(shape[axis]);
        sa.push_back(full_sa[axis]);
        sb.push_back(full_sb[axis]);
      }
      const std::size_t rank {dims.size()};
      // The innermost axis runs as one loop, each operand stepping 1 or 0 (a broadcast one) along it, so a
      // scalar or a row against a matrix is a plain vector loop; the odometer turns only between rows. The rows go
      // by tasks, each task's odometer set from its first row.
      const std::size_t inner          {dims[rank - 1]};
      const std::size_t step_a         {sa[rank - 1]};
      const std::size_t step_b         {sb[rank - 1]};
      const std::size_t rows           {count / inner};
      const std::size_t rows_per_task  {std::max<std::size_t>(1, detail::elementwise_chunk / inner)};
      detail::parallel_for((rows + rows_per_task - 1) / rows_per_task, rows_per_task * inner, [&](std::size_t task) {
                             const std::size_t        first {task * rows_per_task};
                             const std::size_t        last {std::min(first + rows_per_task, rows)};
                             std::vector<std::size_t> index(rank, 0);
                             std::size_t              ia {0};
                             std::size_t              ib {0};
                             for (std::size_t axis {rank - 1}, rest {first}; axis-- > 0;) { // the first row's position
                               index[axis]  = rest % dims[axis];
                               rest        /= dims[axis];
                               ia          += index[axis] * sa[axis];
                               ib          += index[axis] * sb[axis];
                             }
                             // Two axes, short rows, one operand's row the same for every row (stride 0) and the other's rows
                             // together (stepping 1 along a row of two axes, its rows lie `inner` apart): that row repeated into
                             // a run of at least 64, the task's rows one loop against it.
                             if (rank == 2 && inner < 32 && step_a == 1 && step_b == 1 && (sa[0] == 0) != (sb[0] == 0)) {
                               const bool        a_repeats {sa[0] == 0};
                               const T* const    row_of    {a_repeats ? a : b};
                               const T* const    whole     {(a_repeats ? b : a) + (first * inner)};
                               T* const          po        {out.data() + (first * inner)};
                               const std::size_t span      {((64 + inner - 1) / inner) * inner};
                               std::array<T, 96> repeated {};
                               for (std::size_t i {0}; i < span; ++i) {
                                 repeated[i] = row_of[i % inner];
                               }
                               const std::size_t total {(last - first) * inner};
                               for (std::size_t from {0}; from < total; from += span) {
                                 const std::size_t n {std::min(span, total - from)};
                                 if (a_repeats) {
                                   for (std::size_t i {0}; i < n; ++i) {
                                     po[from + i] = op(repeated[i], whole[from + i]);
                                   }
                                 }
                                 else {
                                   for (std::size_t i {0}; i < n; ++i) {
                                     po[from + i] = op(whole[from + i], repeated[i]);
                                   }
                                 }
                               }
                               return;
                             }
                             // Two axes, short rows, one operand a column (one element a row, stepping 0 along it) and the other's
                             // rows together: each run of rows' column elements spread over their rows, one loop against them.
                             if (rank == 2 && inner < 32 && step_a + step_b == 1 && (step_a == 1 ? sb[0] : sa[0]) == 1
                                 && (step_a == 1 ? sa[0] : sb[0]) == inner) {
                               const bool        a_spreads {step_a == 0};
                               const T* const    column    {(a_spreads ? a : b) + first};
                               const T* const    whole     {(a_spreads ? b : a) + (first * inner)};
                               T* const          po        {out.data() + (first * inner)};
                               const std::size_t run       {256 / inner};
                               std::array<T, 256> spread {};
                               for (std::size_t from {0}; from < last - first; from += run) {
                                 const std::size_t n {std::min(run, last - first - from)};
                                 for (std::size_t r {0}; r < n; ++r) {
                                   std::fill_n(spread.data() + (r * inner), inner, column[from + r]);
                                 }
                                 const T* const w {whole + (from * inner)};
                                 T* const       o {po + (from * inner)};
                                 if (a_spreads) {
                                   for (std::size_t i {0}; i < n * inner; ++i) {
                                     o[i] = op(spread[i], w[i]);
                                   }
                                 }
                                 else {
                                   for (std::size_t i {0}; i < n * inner; ++i) {
                                     o[i] = op(w[i], spread[i]);
                                   }
                                 }
                               }
                               return;
                             }
                             for (std::size_t row {first}; row < last; ++row) {
                               const T* const pa {a + ia};
                               const T* const pb {b + ib};
                               T* const       po {out.data() + (row * inner)};
                               if (step_a == 1 && step_b == 1) {
                                 for (std::size_t i {0}; i < inner; ++i) {
                                   po[i] = op(pa[i], pb[i]);
                                 }
                               }
                               else if (step_a == 1) {
                                 const T y {pb[0]};
                                 for (std::size_t i {0}; i < inner; ++i) {
                                   po[i] = op(pa[i], y);
                                 }
                               }
                               else if (step_b == 1) {
                                 const T x {pa[0]};
                                 for (std::size_t i {0}; i < inner; ++i) {
                                   po[i] = op(x, pb[i]);
                                 }
                               }
                               else {
                                 std::fill(po, po + inner, op(pa[0], pb[0]));
                               }
                               for (std::size_t axis {rank - 1}; axis-- > 0;) { // odometer over the outer axes: carry on wrap
                                 ++index[axis];
                                 ia += sa[axis];
                                 ib += sb[axis];
                                 if (index[axis] < dims[axis]) {
                                   break;
                                 }
                                 ia          -= sa[axis] * dims[axis];
                                 ib          -= sb[axis] * dims[axis];
                                 index[axis]  = 0;
                               }
                             }
                           });
      return tensor<T> {std::move(shape), std::move(out)};
    }
  }    // namespace detail

  namespace detail {
    /*!
     * \brief Sum of `count` values from \p first at stride \p stride, by NumPy's pairwise summation: below 8
     *        values in order; up to 128 in eight interleaved partial sums, added pairwise, then the
     *        remainder in order; above, the two halves (the first a multiple of 8 long) summed apart.
     */
    //! \brief The partial sums \ref pairwise_sum interleaves, and the multiple of which its first half is long.
    inline constexpr std::size_t pairwise_lanes {8};

    template <typename T>
    [[nodiscard]] T pairwise_sum(const T*    first,
                                 std::size_t count,
                                 std::size_t stride)
    {
      constexpr std::size_t lanes {pairwise_lanes};
      constexpr std::size_t block {128};
      if (count < lanes) {
        T sum {};
        for (std::size_t i {0}; i < count; ++i) {
          sum += first[i * stride];
        }
        return sum;
      }
      if (count <= block) {
        std::array<T, lanes> partial {};
        for (std::size_t j {0}; j < lanes; ++j) {
          partial[j] = first[j * stride];
        }
        std::size_t i {lanes};
        for (; i < count - (count % lanes); i += lanes) {
          for (std::size_t j {0}; j < lanes; ++j) {
            partial[j] += first[(i + j) * stride];
          }
        }
        T sum {((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]))};
        for (; i < count; ++i) {
          sum += first[i * stride];
        }
        return sum;
      }
      std::size_t half {count / 2};
      half -= half % lanes;
      return pairwise_sum(first, half, stride) + pairwise_sum(first + (half * stride), count - half, stride);
    }

    //! \brief The longest subtree of \ref pairwise_sum's tree \ref parallel_pairwise_sum sums on one thread: 256 KiB
    //!        of `double`.
    inline constexpr std::size_t pairwise_leaf {std::size_t {1} << 15U};

    //! \brief The lengths of \ref pairwise_sum's subtrees of at most \ref pairwise_leaf values over \p count, in order.
    inline void pairwise_leaves(std::size_t               count,
                                std::vector<std::size_t>& out)
    {
      if (count <= pairwise_leaf) {
        out.push_back(count);
        return;
      }
      std::size_t half {count / 2};
      half -= half % pairwise_lanes;
      pairwise_leaves(half, out);
      pairwise_leaves(count - half, out);
    }

    //! \brief The subtrees' sums from \p next on (\ref pairwise_leaves' order) added as \ref pairwise_sum's tree over
    //!        \p count values adds its halves.
    template <typename T>
    [[nodiscard]] T pairwise_combine(std::size_t count,
                                     const T*&   next)
    {
      if (count <= pairwise_leaf) {
        return *next++;
      }
      std::size_t half {count / 2};
      half -= half % pairwise_lanes;
      const T left     {pairwise_combine(half, next)};
      const T right    {pairwise_combine(count - half, next)};
      return left + right;
    }

    /*!
     * \brief \ref pairwise_sum of \p count contiguous values on the threads, to the bit: the tree's subtrees of at most
     *        \ref pairwise_leaf values summed in parallel, each as pairwise_sum sums it, then added in the tree's order.
     *        The subtrees follow from the count alone. Four subtrees or fewer: on the caller (2¹⁶ doubles, ten AVX-512
     *        cores: 11.7 µs alone, 14.5 shared; 2²⁰ 247 → 44 µs, 2²⁴ 12.5 → 2.2 ms).
     */
    template <typename T>
    [[nodiscard]] T parallel_pairwise_sum(const T*    first,
                                          std::size_t count)
    {
      if (count <= 4 * pairwise_leaf) {
        return pairwise_sum(first, count, 1);
      }
      std::vector<std::size_t> lengths;
      pairwise_leaves(count, lengths);
      std::vector<std::size_t> starts(lengths.size());
      std::exclusive_scan(lengths.begin(), lengths.end(), starts.begin(), std::size_t {0});
      buffer<T> sums(lengths.size());
      parallel_for(lengths.size(), pairwise_leaf, [&](std::size_t leaf) { sums[leaf] = pairwise_sum(first + starts[leaf], lengths[leaf], 1); });
      const T* next {sums.data()};
      return pairwise_combine(count, next);
    }

    /*!
     * \brief \ref pairwise_sum of \p width neighbouring columns at once, into \p out: column `q`'s sum that of
     *        `pairwise_sum(first + q, count, stride)`, to the bit, each step adding a whole row of the columns
     *        (contiguous, \p stride apart) where a column was walked down \p stride at a time.
     */
    template <typename T>
    void pairwise_sum_rows(const T*    first,
                           std::size_t count,
                           std::size_t stride,
                           std::size_t width,
                           T*          out)
    {
      constexpr std::size_t lanes {8};
      constexpr std::size_t block {128};
      const auto            row = [&](std::size_t i) { return first + (i * stride); };
      if (count < lanes) {
        std::fill_n(out, width, T {});
        for (std::size_t i {0}; i < count; ++i) {
          for (std::size_t q {0}; q < width; ++q) {
            out[q] += row(i)[q];
          }
        }
        return;
      }
      if (count <= block) {
        std::vector<T> partial(lanes * width);
        for (std::size_t j {0}; j < lanes; ++j) {
          std::copy_n(row(j), width, partial.data() + (j * width));
        }
        std::size_t i {lanes};
        for (; i < count - (count % lanes); i += lanes) {
          for (std::size_t j {0}; j < lanes; ++j) {
            T* const       sums {partial.data() + (j * width)};
            const T* const next {row(i + j)};
            for (std::size_t q {0}; q < width; ++q) {
              sums[q] += next[q];
            }
          }
        }
        const T* const p {partial.data()};
        for (std::size_t q {0}; q < width; ++q) {
          out[q] = ((p[q] + p[width + q]) + (p[(2 * width) + q] + p[(3 * width) + q]))
                   + ((p[(4 * width) + q] + p[(5 * width) + q]) + (p[(6 * width) + q] + p[(7 * width) + q]));
        }
        for (; i < count; ++i) {
          for (std::size_t q {0}; q < width; ++q) {
            out[q] += row(i)[q];
          }
        }
        return;
      }
      std::size_t half {count / 2};
      half -= half % lanes;
      std::vector<T> second(width);
      pairwise_sum_rows(first, half, stride, width, out);
      pairwise_sum_rows(row(half), count - half, stride, width, second.data());
      for (std::size_t q {0}; q < width; ++q) {
        out[q] = out[q] + second[q];
      }
    }

    //! \brief The columns \ref pairwise_sum_rows sums at once: its partial sums of eight rows, 32 KiB of doubles,
    //!        stay in the L1 cache.
    inline constexpr std::size_t pairwise_columns {512};

    /*!
     * \brief The pairwise sums of \p outer blocks of \p count rows of \p inner elements (row-major), each
     *        column of a block summed down its rows, into \p out (`outer × inner`): one column at a time where
     *        \p inner is 1, else by rows (\ref pairwise_sum_rows), the same bits.
     */
    template <typename T>
    void pairwise_sums(const T*    data,
                       std::size_t outer,
                       std::size_t count,
                       std::size_t inner,
                       T*          out)
    {
      for (std::size_t o {0}; o < outer; ++o) {
        const T* const base {data + (o * count * inner)};
        if (inner == 1) {
          out[o] = parallel_pairwise_sum(base, count);
          continue;
        }
        for (std::size_t q {0}; q < inner; q += pairwise_columns) {
          pairwise_sum_rows(base + q, count, inner, std::min(pairwise_columns, inner - q), out + (o * inner) + q);
        }
      }
    }

    //! \brief Is \p ch an ASCII letter (a valid index label)?
    [[nodiscard]] inline bool is_label(char ch)
    {
      return std::isalpha(static_cast<unsigned char>(ch)) != 0;
    }

    //! \brief Splits an einsum left-hand side into per-operand subscripts.
    [[nodiscard]] inline std::vector<std::string> split_inputs(const std::string& lhs)
    {
      std::vector<std::string> groups;
      std::string              current;
      for (const char ch : lhs) {
        if (ch == ',') {
          groups.push_back(current);
          current.clear();
        }
        else if (is_label(ch)) {
          current.push_back(ch);
        }
        else {
          throw einsum_error("einsum: invalid character in input subscripts");
        }
      }
      groups.push_back(current);
      return groups;
    }

    /*!
     * \brief Matrix product `C(M×N) = A(M×K) · B(K×N)`, row-major.
     *
     * The fast path for a plain two-operand single-contraction einsum (matmul, matrix–vector, dot), by
     * \ref detail::gemm. Every element of C is summed in ascending order of the contraction, the
     * general engine's order, so no tolerance changes.
     */
    template <typename T>
    [[nodiscard]] tensor<T> tile_gemm(const T*                 a,
                                      std::size_t              lda,
                                      const T*                 b,
                                      std::size_t              ldb,
                                      std::size_t              rows,
                                      std::size_t              inner,
                                      std::size_t              cols,
                                      std::vector<std::size_t> out_shape)
    {
      buffer<T> c(rows * cols); // the caller routes only an output of two elements or more here
      gemm(rows, cols, inner, a, lda, b, ldb, c.data(), cols, c_start::zero);
      return tensor<T> {std::move(out_shape), std::move(c)};
    }

    //! \brief The columns a thread takes of a vector times a matrix: each of its rows added to them in order.
    inline constexpr std::size_t vector_chunk {128};

    //! \brief The columns of B one task of \ref vector_times_columns sums together: their sums' chains interleaved,
    //!        each waiting on its own last addition.
    inline constexpr std::size_t strided_columns {8};

    //! \brief Columns \p j to `j + Width` of \ref vector_times_columns into \p y, their sums held through the rows.
    template <std::size_t Width, typename T>
    void strided_column_group(const T*    a,
                              const T*    b,
                              std::size_t row_step,
                              std::size_t column_step,
                              std::size_t inner,
                              std::size_t j,
                              T*          y)
    {
      std::array<T, Width> sums {};
      for (std::size_t k {0}; k < inner; ++k) {
        const T        f   {a[k]};
        const T* const row {b + (k * row_step) + (j * column_step)};
        for (std::size_t c {0}; c < Width; ++c) {
          add_one_product(sums[c], f, row[c * column_step]);
        }
      }
      std::copy(sums.begin(), sums.end(), y + j);
    }

    /*!
     * \brief \ref canonical_product's vector \p a (\p inner) times a matrix \p b whose element `B[k][j]` is
     *        `b[(k · row_step) + (j · column_step)]`: each element of the product the sum canonical_product forms,
     *        `a[k] · B[k][j]` added in ascending order of `k` with \ref add_product's arithmetic (\ref
     *        add_one_product), so the same bits; \ref strided_columns columns' sums held at once, where add_product
     *        stores and reloads them for every row of B.
     */
    template <typename T>
    [[nodiscard]] buffer<T> vector_times_columns(const T*    a,
                                                 const T*    b,
                                                 std::size_t row_step,
                                                 std::size_t column_step,
                                                 std::size_t inner,
                                                 std::size_t cols)
    {
      static_assert(strided_columns == 8, "the switch below names each narrower last group");
      buffer<T> y(cols);
      parallel_for((cols + vector_chunk - 1) / vector_chunk, inner * vector_chunk, [&](std::size_t chunk) {
                     const std::size_t last {std::min((chunk + 1) * vector_chunk, cols)};
                     std::size_t j {chunk * vector_chunk};
                     for (; j + strided_columns <= last; j += strided_columns) {
                       strided_column_group<strided_columns>(a, b, row_step, column_step, inner, j, y.data());
                     }
                     switch (last - j) {
                       case 1: strided_column_group<1>(a, b, row_step, column_step, inner, j, y.data()); break;
                       case 2: strided_column_group<2>(a, b, row_step, column_step, inner, j, y.data()); break;
                       case 3: strided_column_group<3>(a, b, row_step, column_step, inner, j, y.data()); break;
                       case 4: strided_column_group<4>(a, b, row_step, column_step, inner, j, y.data()); break;
                       case 5: strided_column_group<5>(a, b, row_step, column_step, inner, j, y.data()); break;
                       case 6: strided_column_group<6>(a, b, row_step, column_step, inner, j, y.data()); break;
                       case 7: strided_column_group<7>(a, b, row_step, column_step, inner, j, y.data()); break;
                       default: break; // a whole number of groups
                     }
                   });
      return y;
    }

#ifdef NUMERICAL_X86_COMPILED
    //! \brief \ref vector_times_columns for `double`: on the x86-64 fused paths, where \ref add_product adds each
    //!        row by fused multiply-adds, by \ref avx2_strided_columns; elsewhere the template's arithmetic.
    [[nodiscard]] inline buffer<double> vector_times_columns(const double* a,
                                                             const double* b,
                                                             std::size_t   row_step,
                                                             std::size_t   column_step,
                                                             std::size_t   inner,
                                                             std::size_t   cols)
    {
      const gemm_path path {gemm_path_current()};
      if (path != gemm_path::avx2 && path != gemm_path::avx512) {
        return vector_times_columns<double>(a, b, row_step, column_step, inner, cols);
      }
      buffer<double> y(cols);
      parallel_for((cols + vector_chunk - 1) / vector_chunk, inner * vector_chunk, [&](std::size_t chunk) {
                     avx2_strided_columns(a, b, row_step, column_step, inner, chunk * vector_chunk, std::min((chunk + 1) * vector_chunk, cols), y.data());
                   });
      return y;
    }

#endif

    //! \brief \ref vector_times_columns of a matrix given as its transpose \p bt (`cols × inner`, its rows \p ldbt
    //!         apart, B's columns).
    template <typename T>
    [[nodiscard]] buffer<T> vector_times_transposed(const T*    a,
                                                    const T*    bt,
                                                    std::size_t ldbt,
                                                    std::size_t inner,
                                                    std::size_t cols)
    {
      return vector_times_columns(a, bt, 1, ldbt, inner, cols);
    }

    /*!
     * \brief The columns under which \ref canonical_product sums a vector times a matrix by \ref vector_times_columns,
     *        the same bits as its rows added in order: below it faster for every type from a depth of a hundred (a few
     *        nanoseconds either way under it), from it slower for a real type. On x86-64 the rows' vectors of eight
     *        `float` (AVX2) make the rows' loop the faster sooner.
     */
#if defined(__aarch64__) || defined(_M_ARM64)
    inline constexpr std::size_t column_sums_below {16};
#else
    inline constexpr std::size_t column_sums_below {9};
#endif

    /*!
     * \brief The product of the row-major `rows × inner` \p a and `inner × cols` \p b, shaped \p out_shape: the
     *        canonical two-operand contraction of \ref einsum_impl (a matrix times a vector, a vector times a
     *        matrix, a matrix product, a dot product). A matrix's rows lie \p lda (\p a) and \p ldb (\p b) apart,
     *        each row's elements together; a vector's elements together (\p b of one column: \p ldb then 1).
     */
    template <typename T>
    [[nodiscard]] tensor<T> canonical_product(const T*                 a,
                                              std::size_t              lda,
                                              const T*                 b,
                                              std::size_t              ldb,
                                              std::size_t              rows,
                                              std::size_t              inner,
                                              std::size_t              cols,
                                              std::vector<std::size_t> out_shape)
    {
      if (cols == 1 && rows > 1 && inner != 0) { // a matrix times a vector: each row's dot product, as "i,i->"
        buffer<T> y(rows);
        matrix_times_vector(a, lda, rows, inner, b, y.data());
        return tensor<T> {std::move(out_shape), std::move(y)};
      }
      if (rows == 1 && cols > 1 && inner != 0) { // a vector times a matrix: its rows scaled and added in order
        if (cols < column_sums_below) {          // a narrow one: its columns' sums held instead, the same bits
          return tensor<T> {std::move(out_shape), vector_times_columns(a, b, ldb, 1, inner, cols)};
        }
        buffer<T> y(cols, T {});
        parallel_for((cols + vector_chunk - 1) / vector_chunk, inner * vector_chunk, [&](std::size_t chunk) {
                       const std::size_t first {chunk * vector_chunk};
                       const std::size_t count {std::min(vector_chunk, cols - first)};
                       for (std::size_t k {0}; k < inner; ++k) {
                         add_product(y.data() + first, a[k], b + (k * ldb) + first, count);
                       }
                     });
        return tensor<T> {std::move(out_shape), std::move(y)};
      }
      if (rows * cols != 1) { // the product, an empty one included; a single element is one dot product
        return tile_gemm<T>(a, lda, b, ldb, rows, inner, cols, std::move(out_shape));
      }
      return tensor<T> {std::move(out_shape), buffer<T> {chunked_dot(a, b, inner)}};
    }

    /*!
     * \brief einsum of one row-major operand \p data (\p shape, its axes labelled \p in, each label once) into the
     *        labels \p output, where those it sums are one run of its axes and \p output the others in order (ij->i,
     *        ij->, ij->j, ijk->ik): each output element the sum of evenly spaced elements, by pairwise halving as sum()
     *        adds them (\ref pairwise_sums), where einsum's general loop added them one after another. None for any
     *        other spec (nothing summed, the summed labels apart, the output reordered).
     */
    template <typename T>
    [[nodiscard]] std::optional<tensor<T>> sum_over_run(const std::string&              in,
                                                        const std::string&              output,
                                                        const std::vector<std::size_t>& shape,
                                                        const T*                        data)
    {
      const std::size_t first {in.find_first_not_of(output)};
      if (first == std::string::npos) {
        return std::nullopt; // nothing summed: a permutation, or the operand itself
      }
      const std::size_t next {in.find_first_of(output, first)};
      const std::size_t past {next == std::string::npos ? in.size() : next};
      if (in.substr(0, first) + in.substr(past) != output) {
        return std::nullopt;
      }
      std::vector<std::size_t> out_shape;
      for (std::size_t k {0}; k < in.size(); ++k) {
        if (k < first || k >= past) {
          out_shape.push_back(shape[k]);
        }
      }
      // The result's elements: none where a kept extent is 0, whatever the others (each part then 0); else their
      // checked product, which bounds the parts, and the run's with them the operand's.
      const std::size_t kept  {element_count(out_shape.begin(), out_shape.end())};
      if (kept == 0) {
        return tensor<T> {std::move(out_shape), buffer<T> {}};
      }
      const auto        begin {shape.begin()};
      const std::size_t outer {element_count(begin, begin + static_cast<std::ptrdiff_t>(first))};
      const std::size_t run   {element_count(begin + static_cast<std::ptrdiff_t>(first), begin + static_cast<std::ptrdiff_t>(past))};
      const std::size_t inner {element_count(begin + static_cast<std::ptrdiff_t>(past), shape.end())};
      std::vector<T>    sums(kept);
      pairwise_sums(data, outer, run, inner, sums.data());
      return tensor<T> {std::move(out_shape), std::move(sums)};
    }

    /*!
     * \brief The elements of \p data, whose axes carry the labels \p from (extents in \p sizes), laid out row-major
     *        along the labels \p to, a permutation of \p from that differs from it (so at least one label): an
     *        odometer over \p to's outer labels, a strided sweep of its last.
     */
    template <typename T>
    [[nodiscard]] buffer<T> permuted(const buffer<T>&                   data,
                                     const std::string&                 from,
                                     const std::string&                 to,
                                     const std::map<char, std::size_t>& sizes)
    {
      buffer<T> out(data.size()); // every element written below
      if (data.empty()) {
        return out;
      }
      if (to.size() == 2) { // the one permutation of two axes, a transpose: by tiles, each read and written in the cache
        const std::size_t rows  {sizes.at(from[0])};
        const std::size_t cols  {sizes.at(from[1])};
        const std::size_t piece {transpose_piece(cols * sizeof(T), rows * sizeof(T))};
        parallel_for((rows + transpose_tile - 1) / transpose_tile, transpose_tile * cols, [&](std::size_t block) {
                       const std::size_t r0 {block * transpose_tile};
                       const std::size_t r1 {std::min(r0 + transpose_tile, rows)};
                       for (std::size_t c0 {0}; c0 < cols; c0 += transpose_tile) {
                         transpose_walk(r0, r1, c0, std::min(c0 + transpose_tile, cols), piece,
                                        [&](std::size_t r, std::size_t c) { out[(c * rows) + r] = data[(r * cols) + c]; });
                       }
                     });
        return out;
      }
      const std::size_t           rank {to.size()};
      std::map<char, std::size_t> source; // each label's stride in data
      std::size_t                 step {1};
      for (std::size_t axis {from.size()}; axis-- > 0;) {
        source[from[axis]]  = step;
        step               *= sizes.at(from[axis]);
      }
      std::vector<std::size_t> stride(rank);
      std::vector<std::size_t> extent(rank);
      for (std::size_t axis {0}; axis < rank; ++axis) {
        stride[axis] = source.at(to[axis]);
        extent[axis] = sizes.at(to[axis]);
      }
      const std::size_t        inner  {extent[rank - 1]};
      const std::size_t        sweep  {stride[rank - 1]};
      std::vector<std::size_t> index(rank - 1, 0);
      std::size_t              offset {0};
      for (std::size_t row {0}; row < data.size() / inner; ++row) {
        for (std::size_t i {0}; i < inner; ++i) {
          out[(row * inner) + i] = data[offset + (i * sweep)];
        }
        for (std::size_t axis {rank - 1}; axis-- > 0;) {
          offset += stride[axis];
          if (++index[axis] < extent[axis]) {
            break;
          }
          offset      -= stride[axis] * extent[axis];
          index[axis]  = 0;
        }
      }
      return out;
    }

    /*!
     * \brief A two-operand product that sums nothing (`ij,j->ij`, `i,j->ij`), each element of the output its operands'
     *        product added to 0 as \ref gemm_rows rounds a product's first term (\ref rows_terms_fuse), the bits of
     *        \ref contract_pair's products of depth one: each operand's axes in the output's order (a copy only where
     *        they lie otherwise), broadcast against the other (\ref broadcast_into), where those products went one
     *        matrix product of depth one per element of the batch (1000 × 1000 times a row of 1000, six AVX2 cores:
     *        19.7 → 0.5 ms).
     */
    /*!
     * \brief `std::fma(p, q, +0)` without the instruction (in code built for any x86-64, `std::fma` is a call to the
     *        library, 40 000 elements of an outer product 18 → 114 µs on four AVX2 cores): the rounded product, which
     *        keeps the sign of a product that underflows to zero, plus +0 where an operand is zero, as the fused sum
     *        of an exact zero product and +0 is.
     */
    template <std::floating_point T>
    [[nodiscard]] inline T fused_from_zero(T p,
                                           T q) noexcept
    {
      const T product {p * q};
      return p == T {} || q == T {} ? product + T {} : product;
    }

    template <typename T>
    [[nodiscard]] tensor<T> each_product(const std::string&                 sa,
                                         const tensor<T>&                   a,
                                         const std::string&                 sb,
                                         const tensor<T>&                   b,
                                         const std::string&                 output,
                                         const std::map<char, std::size_t>& sizes)
    {
      const auto in_output_order {[&](const std::string& labels, const tensor<T>& t, buffer<T>& copy, std::vector<std::size_t>& shape) {
                                    std::string order;
                                    for (const char label : output) {
                                      const bool carried {labels.find(label) != std::string::npos};
                                      shape.push_back(carried ? sizes.at(label) : 1);
                                      if (carried) {
                                        order.push_back(label);
                                      }
                                    }
                                    if (order == labels) {
                                      return t.data().data();
                                    }
                                    copy = permuted(t.data(), labels, order, sizes);
                                    return static_cast<const T*>(copy.data());
                                  }};
      buffer<T>                a_copy;
      buffer<T>                b_copy;
      std::vector<std::size_t> a_shape;
      std::vector<std::size_t> b_shape;
      const T* const           x {in_output_order(sa, a, a_copy, a_shape)};
      const T* const           y {in_output_order(sb, b, b_copy, b_shape)};
      if constexpr (std::is_floating_point_v<T>) { // gemm_rows fuses no integer's terms
        if (rows_terms_fuse<T>()) {
          return broadcast_into(x, a_shape, y, b_shape, [](T p, T q) { return fused_from_zero(p, q); });
        }
      }
      return broadcast_into(x, a_shape, y, b_shape, [](T p, T q) { return rounded_apart_add(T {}, p, q); });
    }

    /*!
     * \brief A two-operand contraction as matrix products: the labels both operands and the output carry (batch),
     *        those one operand and the output carry (free), and those both operands carry alone (summed, in
     *        ascending order, the order the general loop sums them in, so each element takes the same terms in the
     *        same order) regroup each operand as a stack of matrices, `A` as `batch × free_a × summed` and `B` as
     *        `batch × summed × free_b`, multiplied batch by batch through the packed kernel, batches in parallel;
     *        the result is laid out in the output's order; a dot product per batch by \ref dot_product. Nothing for a
     *        label repeated within an operand or summed by one operand alone (the general loop takes those).
     */
    template <typename T>
    [[nodiscard]] std::optional<tensor<T>> contract_pair(const std::string&                 sa,
                                                         const tensor<T>&                   a,
                                                         const std::string&                 sb,
                                                         const tensor<T>&                   b,
                                                         const std::string&                 output,
                                                         const std::string&                 summed,
                                                         const std::map<char, std::size_t>& sizes)
    {
      const auto repeats {[](const std::string& labels) {
                            for (std::size_t i {0}; i < labels.size(); ++i) {
                              if (labels.find(labels[i], i + 1) != std::string::npos) {
                                return true;
                              }
                            }
                            return false;
                          }};
      if (repeats(sa) || repeats(sb)) {
        return std::nullopt;
      }
      std::string batch;
      std::string free_a;
      std::string free_b;
      for (const char label : output) {
        const bool in_a {sa.find(label) != std::string::npos};
        const bool in_b {sb.find(label) != std::string::npos};
        (in_a && in_b ? batch : (in_a ? free_a : free_b)).push_back(label);
      }
      if (batch + free_a + free_b != output && batch + free_b + free_a == output) {
        // The other operand first lays the result out in the output's order: the same terms in the same order
        // (each a product of the same two numbers), with no permutation of the result.
        return contract_pair(sb, b, sa, a, output, summed, sizes);
      }
      std::size_t inner {1};
      for (const char label : summed) {
        if (sa.find(label) == std::string::npos || sb.find(label) == std::string::npos) {
          return std::nullopt;
        }
        inner *= sizes.at(label);
      }
      if constexpr (std::is_arithmetic_v<T>) {
        if (summed.empty() && !(free_a.empty() && free_b.empty())) {
          return each_product(sa, a, sb, b, output, sizes);
        }
      }
      const auto extent_of {[&sizes](const std::string& labels) {
                              std::size_t product {1};
                              for (const char label : labels) {
                                product = checked_mul(product, sizes.at(label));
                              }
                              return product;
                            }};
      const std::size_t    batches                                            {extent_of(batch)};
      const std::size_t    rows                                               {extent_of(free_a)};
      const std::size_t    cols                                               {extent_of(free_b)};
      const std::string    left_order                                         {batch + free_a + summed};
      const std::string    right_order                                        {batch + summed + free_b};
      const buffer<T>      left                                               {sa == left_order ? buffer<T> {} : permuted(a.data(), sa, left_order, sizes)};
      const buffer<T>      right                                              {sb == right_order ? buffer<T> {} : permuted(b.data(), sb, right_order, sizes)};
      const T* const       left_data                                          {sa == left_order ? a.data().data() : left.data()};
      const T* const       right_data                                         {sb == right_order ? b.data().data() : right.data()};
      buffer<T>            c(checked_mul(batches, checked_mul(rows, cols)));
      if (rows == 1 && cols == 1) { // a dot product per batch, as `chunked_dot` sums each chunk
        parallel_for(batches, inner, [&](std::size_t p) {
                       c[p] = dot_product(left_data + (p * inner), right_data + (p * inner), inner);
                     });
      }
      else {
        const auto product {[&](std::size_t p) {
                              gemm(rows, cols, inner, left_data + (p * rows * inner), inner, right_data + (p * inner * cols), cols,
                                   c.data() + (p * rows * cols), cols, c_start::zero);
                            }};
        if (batches < max_threads()) { // fewer than the threads: one after another, each on all of them
          for (std::size_t p {0}; p < batches; ++p) {
            product(p);
          }
        }
        else {
          parallel_for(batches, rows * cols * inner / gemm_kernel_speedup, product);
        }
      }
      std::vector<std::size_t> shape;
      for (const char label : output) {
        shape.push_back(sizes.at(label));
      }
      const std::string order {batch + free_a + free_b};
      return tensor<T> {std::move(shape), order == output ? std::move(c) : permuted(c, order, output, sizes)};
    }

    //! \brief The operands at most whose contraction order \ref contraction_path searches exhaustively; more take
    //!        the cheapest pair at each step.
    inline constexpr std::size_t exhaustive_path_operands {4};

    //! \brief One step of a contraction path: the operands at \p first and \p second of the current list, whose
    //!        result goes to the list's end.
    struct contraction_step
    {
      std::size_t first;  //!< The first operand's position.
      std::size_t second; //!< The second's, past the first.
    };

    //! \brief The labels of \p labels[\p i] and \p labels[\p j] that the output or another operand carries, in their
    //!        order of appearance, once each: what their contraction keeps.
    [[nodiscard]] inline std::string pair_keeps(const std::vector<std::string>& labels,
                                                std::size_t                     i,
                                                std::size_t                     j,
                                                const std::string&              output)
    {
      std::string elsewhere {output};
      for (std::size_t k {0}; k < labels.size(); ++k) {
        elsewhere += k == i || k == j ? std::string {} : labels[k];
      }
      std::string keep;
      for (const char label : labels[i] + labels[j]) {
        if (elsewhere.find(label) != std::string::npos && keep.find(label) == std::string::npos) {
          keep.push_back(label);
        }
      }
      return keep;
    }

    //! \brief The product of the extents of \p labels' distinct labels, as a `double` (a cost: it may pass `size_t`).
    [[nodiscard]] inline double extent_product(const std::string&                 labels,
                                               const std::map<char, std::size_t>& sizes)
    {
      double      product {1.0};
      std::string seen;
      for (const char label : labels) {
        if (seen.find(label) == std::string::npos) {
          seen.push_back(label);
          product *= static_cast<double>(sizes.at(label));
        }
      }
      return product;
    }

    //! \brief The list after a contraction step: the two operands' labels removed, their result's appended.
    [[nodiscard]] inline std::vector<std::string> after_step(const std::vector<std::string>& labels,
                                                             contraction_step                step,
                                                             const std::string&              output)
    {
      std::vector<std::string> next;
      for (std::size_t k {0}; k < labels.size(); ++k) {
        if (k != step.first && k != step.second) {
          next.push_back(labels[k]);
        }
      }
      next.push_back(pair_keeps(labels, step.first, step.second, output));
      return next;
    }

    //! \brief The cheapest order of contraction from \p labels, and its cost and largest intermediate: every order
    //!        tried, the first found kept among equals.
    inline void cheapest_path(const std::vector<std::string>&    labels,
                              const std::string&                 output,
                              const std::map<char, std::size_t>& sizes,
                              std::vector<contraction_step>&     path,
                              double&                            cost,
                              double&                            largest)
    {
      if (labels.size() < 2) {
        path.clear();
        cost    = 0.0;
        largest = 0.0;
        return;
      }
      bool found {false};
      for (std::size_t i {0}; i < labels.size(); ++i) {
        for (std::size_t j {i + 1}; j < labels.size(); ++j) {
          const contraction_step        step         {.first = i, .second = j};
          std::vector<contraction_step> rest;
          double                        rest_cost    {0.0};
          double                        rest_largest {0.0};
          cheapest_path(after_step(labels, step, output), output, sizes, rest, rest_cost, rest_largest);
          const double here                          {extent_product(labels[i] + labels[j], sizes)};
          const double size                          {extent_product(pair_keeps(labels, i, j, output), sizes)};
          const double total                         {here + rest_cost};
          const double peak                          {std::max(size, rest_largest)};
          if (!found || total < cost || (total == cost && peak < largest)) {
            found   = true;
            cost    = total;
            largest = peak;
            path.assign(1, step);
            path.insert(path.end(), rest.begin(), rest.end());
          }
        }
      }
    }

    /*!
     * \brief The order in which to contract operands labelled \p labels into \p output, as `opt_einsum` chooses it:
     *        the cheapest in multiply-adds (the product of the extents a pair's labels span), the smaller largest
     *        intermediate among equals, over every order up to \ref exhaustive_path_operands operands; past it, the
     *        cheapest pair at each step among those sharing a label (an outer product, cheap in itself, would leave
     *        its labels to every later step), the smaller result, then the first pair, among equals. It depends on
     *        the shapes alone, so a result is the same on any number of threads.
     */
    [[nodiscard]] inline std::vector<contraction_step> contraction_path(std::vector<std::string>           labels,
                                                                        const std::string&                 output,
                                                                        const std::map<char, std::size_t>& sizes)
    {
      std::vector<contraction_step> path;
      if (labels.size() <= exhaustive_path_operands) {
        double cost    {0.0};
        double largest {0.0};
        cheapest_path(labels, output, sizes, path, cost, largest);
        return path;
      }
      const auto shares {[&labels](std::size_t i, std::size_t j) {
                           return labels[i].find_first_of(labels[j]) != std::string::npos;
                         }};
      while (labels.size() > 1) {
        contraction_step best      {.first = 0, .second = 1};
        double           best_cost {-1.0};
        double           best_size {0.0};
        bool             connected {false}; // a pair sharing a label exists: an outer product waits
        for (std::size_t i {0}; i < labels.size(); ++i) {
          for (std::size_t j {i + 1}; j < labels.size(); ++j) {
            connected = connected || shares(i, j);
          }
        }
        for (std::size_t i {0}; i < labels.size(); ++i) {
          for (std::size_t j {i + 1}; j < labels.size(); ++j) {
            if (connected && !shares(i, j)) {
              continue;
            }
            const double here {extent_product(labels[i] + labels[j], sizes)};
            const double size {extent_product(pair_keeps(labels, i, j, output), sizes)};
            if (best_cost < 0.0 || here < best_cost || (here == best_cost && size < best_size)) {
              best      = {.first = i, .second = j};
              best_cost = here;
              best_size = size;
            }
          }
        }
        path.push_back(best);
        labels = after_step(labels, best, output);
      }
      return path;
    }

    //! \brief Evaluates a parsed einsum over its operands. \see numerical::einsum
    //!
    //! After parsing and the consistency checks, the contraction runs over a
    //! single integer index space — the output (free) labels first, then the
    //! summed labels. Each operand's flat offset is carried incrementally by a
    //! mixed-radix odometer using a per-label stride coefficient computed once,
    //! so the hot loop allocates nothing and performs no map lookups; its
    //! innermost reduction is a strided accumulation the compiler can
    //! auto-vectorise. (Pairwise contraction-path optimisation, as in opt_einsum,
    //! and a BLAS reduction are deliberately not included — they would trade the
    //! library's generic simplicity and zero dependencies for raw throughput.)
    //! \brief The position of the ellipsis in \p subscripts, or `npos` where there is none: `'.'` only as one `...`.
    //! \throws einsum_error If a `'.'` stands otherwise.
    [[nodiscard]] inline std::size_t ellipsis_in(const std::string& subscripts)
    {
      const std::size_t at   {subscripts.find("...")};
      const auto        dots {static_cast<std::size_t>(std::count(subscripts.begin(), subscripts.end(), '.'))};
      if (dots != (at == std::string::npos ? 0U : 3U)) {
        throw einsum_error("einsum: '.' stands only as one ellipsis, '...', per subscript");
      }
      return at;
    }

    template <typename T>
    [[nodiscard]] tensor<T> einsum_impl(const std::string&                   spec_in,
                                        const std::vector<const tensor<T>*>& operands);

    /*!
     * \brief \ref einsum_impl of a spec with ellipses, as NumPy reads them: each `...` stands for an operand's axes its
     *        letters leave, aligned from the right across the operands and broadcast (an extent of 1 repeated), named by
     *        letters the spec does not use. An operand's axis of extent 1 under the ellipsis where another's is longer is
     *        dropped with its letter, so that operand repeats along it. The output: its `...` the broadcast axes (an
     *        output without one, where an operand has such axes, is refused, as NumPy refuses it); with no `->`, the
     *        broadcast axes then the letters that appear once, in order.
     */
    template <typename T>
    [[nodiscard]] tensor<T> einsum_ellipsis(const std::string&                   spec,
                                            const std::vector<const tensor<T>*>& operands)
    {
      const std::size_t        arrow  {spec.find("->")};
      std::vector<std::string> inputs {""};
      for (const char ch : spec.substr(0, arrow)) {
        if (ch == ',') {
          inputs.emplace_back();
        }
        else {
          inputs.back().push_back(ch);
        }
      }
      if (inputs.size() != operands.size()) {
        throw einsum_error("einsum: number of operands does not match the spec");
      }
      std::vector<std::size_t> at(inputs.size());
      std::vector<std::size_t> under(inputs.size(), 0); // the axes each operand's ellipsis stands for
      std::size_t              widest {0};
      for (std::size_t i {0}; i < inputs.size(); ++i) {
        at[i] = ellipsis_in(inputs[i]);
        if (at[i] != std::string::npos) {
          const std::size_t letters {inputs[i].size() - 3};
          if (operands[i]->rank() < letters) {
            throw einsum_error("einsum: operand rank does not match its subscripts");
          }
          under[i] = operands[i]->rank() - letters;
          widest   = std::max(widest, under[i]);
        }
      }
      static constexpr std::string_view alphabet {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
      std::string                       fresh;
      for (const char letter : alphabet) {
        if (fresh.size() < widest && spec.find(letter) == std::string::npos) {
          fresh.push_back(letter);
        }
      }
      if (fresh.size() < widest) {
        throw einsum_error("einsum: too many axes under the ellipsis for the letters left");
      }
      std::vector<std::size_t> extent(widest, 1);
      for (std::size_t i {0}; i < inputs.size(); ++i) {
        for (std::size_t d {0}; d < under[i]; ++d) {
          const std::size_t dim       {operands[i]->shape()[at[i] + d]};
          std::size_t&      broadcast {extent[widest - under[i] + d]};
          if (dim != 1 && broadcast != 1 && dim != broadcast) {
            throw dimension_error("einsum: the axes under the ellipsis do not broadcast");
          }
          broadcast = dim == 1 ? broadcast : dim;
        }
      }
      std::deque<tensor<T>>         owned; // the operands with axes dropped (a deque keeps their addresses)
      std::vector<const tensor<T>*> next;
      std::string                   rewritten;
      for (std::size_t i {0}; i < inputs.size(); ++i) {
        rewritten += i == 0 ? "" : ",";
        if (at[i] == std::string::npos) {
          rewritten += inputs[i];
          next.push_back(operands[i]);
          continue;
        }
        std::string              labels {inputs[i].substr(0, at[i])};
        std::vector<std::size_t> shape(operands[i]->shape().begin(), operands[i]->shape().begin() + static_cast<std::ptrdiff_t>(at[i]));
        for (std::size_t d {0}; d < under[i]; ++d) {
          const std::size_t axis {widest - under[i] + d};
          const std::size_t dim  {operands[i]->shape()[at[i] + d]};
          if (dim != extent[axis]) { // 1 where another operand's is longer: repeated, its letter dropped
            continue;
          }
          labels.push_back(fresh[axis]);
          shape.push_back(dim);
        }
        labels += inputs[i].substr(at[i] + 3);
        shape.insert(shape.end(), operands[i]->shape().begin() + static_cast<std::ptrdiff_t>(at[i] + under[i]), operands[i]->shape().end());
        rewritten += labels;
        if (shape.size() == operands[i]->rank()) {
          next.push_back(operands[i]);
        }
        else {
          owned.emplace_back(std::move(shape), buffer<T>(operands[i]->data().begin(), operands[i]->data().end()));
          next.push_back(&owned.back());
        }
      }
      std::string output;
      if (arrow != std::string::npos) {
        output                  = spec.substr(arrow + 2);
        const std::size_t where {ellipsis_in(output)};
        if (where == std::string::npos && widest > 0) {
          throw einsum_error("einsum: the output keeps the axes under the ellipsis: '...' in it, as NumPy requires");
        }
        if (where != std::string::npos) {
          output.replace(where, 3, fresh);
        }
      }
      else { // implicit: the broadcast axes, then the letters that appear once, in order
        std::map<char, std::size_t> seen;
        for (const char label : spec) {
          if (is_label(label)) {
            ++seen[label];
          }
        }
        output = fresh;
        for (const auto& [label, count] : seen) {
          if (count == 1) {
            output.push_back(label);
          }
        }
      }
      return einsum_impl<T>(rewritten + "->" + output, next);
    }

    template <typename T>
    [[nodiscard]] tensor<T> einsum_impl(const std::string&                    spec_in,
                                        const std::vector<const tensor<T>*>&  operands)
    {
      // Tolerate whitespace anywhere in the spec, the way NumPy does ("ij, jk -> ik").
      std::string spec {spec_in};
      spec.erase(std::remove_if(spec.begin(), spec.end(),
                                [](unsigned char ch) { return std::isspace(ch) != 0; }),
                 spec.end());
      if (spec.find('.') != std::string::npos) {
        return einsum_ellipsis<T>(spec, operands);
      }
      if (spec.find("->") == std::string::npos) {
        // Implicit mode, as NumPy reads it: the output is every label that appears once, in ASCII order (`"ij,jk"`
        // is `"ij,jk->ik"`, `"ii"` the trace, `"ba"` the transpose).
        std::map<char, std::size_t> seen;
        for (const char label : spec) {
          if (label != ',') {
            ++seen[label];
          }
        }
        spec += "->";
        for (const auto& [label, count] : seen) {
          if (count == 1) {
            spec += label;
          }
        }
      }
      const std::size_t              arrow  {spec.find("->")};
      const std::vector<std::string> inputs {split_inputs(spec.substr(0, arrow))};
      const std::string              rhs    {spec.substr(arrow + 2)};

      if (operands.size() != inputs.size()) {
        throw einsum_error("einsum: number of operands does not match the spec");
      }

      // Each label's extent, checked for consistency across every axis it labels.
      std::map<char, std::size_t> sizes;
      for (std::size_t i {0}; i < operands.size(); ++i) {
        if (inputs[i].size() != operands[i]->rank()) {
          throw einsum_error("einsum: operand rank does not match its subscripts");
        }
        for (std::size_t axis {0}; axis < inputs[i].size(); ++axis) {
          const char        label {inputs[i][axis]};
          const std::size_t dim   {operands[i]->shape()[axis]};
          const auto        found {sizes.find(label)};
          if (found == sizes.end()) {
            sizes.emplace(label, dim);
          }
          else if (found->second != dim) {
            throw dimension_error("einsum: inconsistent extent for a repeated label");
          }
        }
      }

      // Output (free) labels: each must be present in the inputs and appear once.
      std::string output;
      for (const char ch : rhs) {
        if (!is_label(ch)) {
          throw einsum_error("einsum: invalid character in output subscripts");
        }
        if (sizes.find(ch) == sizes.end()) {
          throw einsum_error("einsum: output label not present in any input");
        }
        if (output.find(ch) != std::string::npos) {
          throw einsum_error("einsum: repeated output label");
        }
        output.push_back(ch);
      }

      // Summed labels: those that do not survive into the output (sorted, by map).
      std::string summed;
      for (const auto& entry : sizes) {
        if (output.find(entry.first) == std::string::npos) {
          summed.push_back(entry.first);
        }
      }

      // Three operands or more: first sum each one over the labels no other operand and not the output
      // carries, then contract them two at a time in the cheapest order (\ref contraction_path), each
      // intermediate keeping only the labels still to come. One loop over every label of `ij,jk,kl->il` runs n⁴
      // terms; pairs run 2·n³, and `i,j,k,l,m->` over 65 536 ones each runs five sums of 65 536 instead of 2⁸⁰.
      if (operands.size() > 2) {
        std::deque<tensor<T>>         owned; // the reductions and intermediates (a deque keeps their addresses)
        std::vector<const tensor<T>*> current;
        std::vector<std::string>      kept;
        for (std::size_t i {0}; i < operands.size(); ++i) {
          std::string elsewhere {output};
          for (std::size_t j {0}; j < operands.size(); ++j) {
            elsewhere += j == i ? std::string {} : inputs[j];
          }
          std::string keep;
          for (const char label : inputs[i]) {
            if (elsewhere.find(label) != std::string::npos && keep.find(label) == std::string::npos) {
              keep.push_back(label);
            }
          }
          if (keep == inputs[i]) { // nothing of its own to sum: the operand itself
            current.push_back(operands[i]);
          }
          else {
            const std::vector<const tensor<T>*> alone {operands[i]};
            owned.push_back(einsum_impl<T>(inputs[i] + "->" + keep, alone));
            current.push_back(&owned.back());
          }
          kept.push_back(std::move(keep));
        }
        for (const contraction_step step : contraction_path(kept, output, sizes)) {
          const std::string                   keep {kept.size() == 2 ? output : pair_keeps(kept, step.first, step.second, output)};
          const std::vector<const tensor<T>*> pair {current[step.first], current[step.second]};
          owned.push_back(einsum_impl<T>(kept[step.first] + "," + kept[step.second] + "->" + keep, pair));
          current.erase(current.begin() + static_cast<std::ptrdiff_t>(step.second));
          current.erase(current.begin() + static_cast<std::ptrdiff_t>(step.first));
          kept.erase(kept.begin() + static_cast<std::ptrdiff_t>(step.second));
          kept.erase(kept.begin() + static_cast<std::ptrdiff_t>(step.first));
          current.push_back(&owned.back());
          kept.push_back(keep);
        }
        return std::move(owned.back()); // the last contraction's, the output
      }

      // Fast path: one operand, nothing summed, each label once: a permutation of its axes (a transpose), a copy.
      if (operands.size() == 1 && summed.empty() && inputs[0].size() == output.size()) {
        std::vector<std::size_t> out_shape;
        for (const char label : output) {
          out_shape.push_back(sizes.at(label));
        }
        return tensor<T> {std::move(out_shape), inputs[0] == output ? operands[0]->data() : permuted(operands[0]->data(), inputs[0], output, sizes)};
      }

      // Fast path: one operand, each label once, its summed labels one run of its axes and the output the others in
      // order (ij->i, ij->, ij->j, ijk->ik): sum_over_run.
      if (operands.size() == 1 && sizes.size() == inputs[0].size()) {
        if (std::optional<tensor<T>> sums {sum_over_run(inputs[0], output, operands[0]->shape(), operands[0]->data().data())}) {
          return *std::move(sums);
        }
      }

      // Fast path: a plain two-operand single contraction in canonical layout —
      // `A` ending with the contracted label, `B` starting with it, and the
      // output being A's leading free labels then B's trailing ones — is exactly
      // a matrix product, routed to the packed kernel (gemm.hpp). (Any stray copy of
      // the contracted label leaves it in a "free" part, so the output would
      // carry a summed label and this equality fails — correctly declining it.)
      if (operands.size() == 2 && summed.size() == 1) {
        const char         contracted {summed.front()};
        const std::string& s0         {inputs[0]};
        const std::string& s1         {inputs[1]};
        if (!s0.empty() && !s1.empty() && s0.back() == contracted && s1.front() == contracted
            && output == s0.substr(0, s0.size() - 1) + s1.substr(1)) {
          const std::size_t        inner {sizes.at(contracted)};
          std::vector<std::size_t> out_shape;
          for (std::size_t k {0}; k + 1 < s0.size(); ++k) {
            out_shape.push_back(sizes.at(s0[k]));
          }
          for (std::size_t k {1}; k < s1.size(); ++k) {
            out_shape.push_back(sizes.at(s1[k]));
          }
          // The result's elements, 0 where one of its extents is 0 whatever the others (an empty contraction's
          // operands may hold any number along their other axes); else checked, and the rows and columns within.
          if (element_count(out_shape.begin(), out_shape.end()) == 0) {
            return tensor<T> {std::move(out_shape), buffer<T> {}};
          }
          const auto        split {out_shape.begin() + static_cast<std::ptrdiff_t>(s0.size() - 1)};
          const std::size_t rows  {element_count(out_shape.begin(), split)};
          const std::size_t cols  {element_count(split, out_shape.end())};
          return canonical_product<T>(operands[0]->data().data(), inner, operands[1]->data().data(), cols, rows, inner, cols, std::move(out_shape));
        }
      }
      if (operands.size() == 2) {
        if (std::optional<tensor<T>> paired {contract_pair<T>(inputs[0], *operands[0], inputs[1], *operands[1], output, summed, sizes)}) {
          return *std::move(paired);
        }
      }

      // Every distinct label, output (free) first then summed: this fixes the
      // index space and the row-major order of the result.
      const std::string labels   {output + summed};
      const std::size_t n_labels {labels.size()};
      const std::size_t n_free   {output.size()};
      const std::size_t n_ops    {operands.size()};

      std::vector<std::size_t> extent(n_labels);
      for (std::size_t k {0}; k < n_labels; ++k) {
        extent[k] = sizes.at(labels[k]);
      }

      // coef[i][k]: operand i's flat-offset change per unit of label k — the sum
      // of its row-major strides over the axes carrying that label (so a label
      // repeated within one operand, e.g. "ii", walks that operand's diagonal).
      std::vector<std::vector<std::size_t>> coef(n_ops, std::vector<std::size_t>(n_labels, 0));
      for (std::size_t i {0}; i < n_ops; ++i) {
        const std::vector<std::size_t>& shape  {operands[i]->shape()};
        std::size_t                     stride {1};
        for (std::size_t axis {inputs[i].size()}; axis-- > 0;) {
          coef[i][labels.find(inputs[i][axis])]  += stride;
          stride                                 *= shape[axis];
        }
      }

      // Fast path: peel the innermost label of each group — the last free label
      // and the last summed label — into a tight unit-of-stride inner loop, and
      // run the odometer only over the *remaining* (outer) labels. The innermost
      // axis is then a contiguous sweep amortising one odometer step over its
      // whole extent: a strided write for a transpose/broadcast, a strided
      // multiply-accumulate for a contraction — both auto-vectoriser-friendly.
      const bool        has_free_inner {n_free > 0};
      const bool        has_sum_inner  {n_labels > n_free};
      const std::size_t inner_free     {has_free_inner ? extent[n_free - 1] : 1};
      const std::size_t inner_sum      {has_sum_inner ? extent[n_labels - 1] : 1};
      const std::size_t free_outer_end {has_free_inner ? n_free - 1 : 0};
      const std::size_t sum_outer_end  {has_sum_inner ? n_labels - 1 : n_free};

      // Per-operand stride of each peeled axis (0 when that axis is absent).
      std::vector<std::size_t> free_step(n_ops, 0);
      std::vector<std::size_t> sum_step(n_ops, 0);
      for (std::size_t i {0}; i < n_ops; ++i) {
        if (has_free_inner) {
          free_step[i] = coef[i][n_free - 1];
        }
        if (has_sum_inner) {
          sum_step[i] = coef[i][n_labels - 1];
        }
      }

      std::size_t out_size {1};
      for (std::size_t k {0}; k < n_free; ++k) {
        out_size = detail::checked_mul(out_size, extent[k]); // guard the result allocation size
      }
      std::vector<std::size_t> out_shape(extent.begin(),
                                         extent.begin() + static_cast<std::ptrdiff_t>(n_free));
      if (out_size == 0) { // an empty result computes nothing, whatever its other extents multiply to
        return tensor<T> {std::move(out_shape), buffer<T> {}};
      }
      std::size_t out_outer {1}; // bounded by out_size
      for (std::size_t k {0}; k < free_outer_end; ++k) {
        out_outer *= extent[k];
      }
      std::size_t sum_outer {1};
      for (std::size_t k {n_free}; k < sum_outer_end; ++k) {
        sum_outer = detail::checked_mul(sum_outer, extent[k]); // a wrapped count would sum nothing, silently
      }

      std::vector<std::size_t> index(n_labels, 0); // value of each outer label
      std::vector<std::size_t> offset(n_ops, 0);   // each operand's flat offset (outer labels only)
      buffer<T>                result;
      result.reserve(out_size);

      for (std::size_t of {0}; of < out_outer; ++of) {
        for (std::size_t fe {0}; fe < inner_free; ++fe) {
          T acc {};
          for (std::size_t os {0}; os < sum_outer; ++os) {
            for (std::size_t se {0}; se < inner_sum; ++se) {
              // The first factor itself, then the others: a complex 1 times an element was no identity (0 · inf
              // is a NaN, and −0 − (−0) a +0), so a trace or a sum of complex infinities or signed zeros went wrong.
              T prod {operands[0]->data()[offset[0] + (fe * free_step[0]) + (se * sum_step[0])]};
              for (std::size_t i {1}; i < n_ops; ++i) {
                prod *= operands[i]->data()[offset[i] + (fe * free_step[i]) + (se * sum_step[i])];
              }
              acc += prod;
            }
            // Step the outer-summed odometer (labels [n_free, sum_outer_end)).
            for (std::size_t k {sum_outer_end}; k-- > n_free;) {
              if (++index[k] < extent[k]) {
                for (std::size_t i {0}; i < n_ops; ++i) {
                  offset[i] += coef[i][k];
                }
                break;
              }
              index[k] = 0;
              for (std::size_t i {0}; i < n_ops; ++i) {
                offset[i] -= coef[i][k] * (extent[k] - 1);
              }
            }
          }
          result.push_back(acc);
        }
        // Step the outer-free odometer (labels [0, free_outer_end)).
        for (std::size_t k {free_outer_end}; k-- > 0;) {
          if (++index[k] < extent[k]) {
            for (std::size_t i {0}; i < n_ops; ++i) {
              offset[i] += coef[i][k];
            }
            break;
          }
          index[k] = 0;
          for (std::size_t i {0}; i < n_ops; ++i) {
            offset[i] -= coef[i][k] * (extent[k] - 1);
          }
        }
      }
      return tensor<T> {std::move(out_shape), std::move(result)};
    }
  } // namespace detail

  /*!
   * \brief Einstein summation over one or more tensor operands.
   *
   * Evaluates \p spec — input subscripts separated by `,`, then `->`, then the
   * output subscripts — against the operands. Examples: `"ij,jk->ik"` (matrix
   * product), `"i,i->"` (dot product), `"ij->ji"` (transpose), `"ii->"` (trace),
   * `"i,j->ij"` (outer product), `"ij->i"` (row sums).
   *
   * \param[in] spec  The einsum specification.
   * \param[in] first The first operand (fixes the scalar type).
   * \param[in] rest  Any further operands.
   * \return The resulting tensor (rank-0 when the output subscripts are empty).
   * \throws einsum_error    If the spec is malformed, the operand count or a rank
   *                         disagrees with it, or an output label is invalid.
   * \throws dimension_error If a shared label labels axes of differing extents.
   */
  template <typename T, typename ... Rest>
  [[nodiscard]] tensor<T> einsum(const std::string& spec,
                                 const tensor<T>&   first,
                                 const Rest& ...    rest)
  {
    const std::vector<const tensor<T>*> operands {&first, (&rest)...};
    return detail::einsum_impl<T>(spec, operands);
  }

  /*!
   * \brief Einstein summation over operands counted at run time: \ref einsum for a caller that holds
   *        them in a container (an interpreter evaluating `einsum("ij,jk->ik", a, b)` from a parse).
   * \param[in] spec     The einsum specification.
   * \param[in] operands The operands, in the spec's order (none null).
   * \return The resulting tensor.
   * \throws einsum_error    As \ref einsum, and if \p operands is empty.
   * \throws dimension_error As \ref einsum.
   */
  template <typename T>
  [[nodiscard]] tensor<T> einsum(const std::string&                   spec,
                                 const std::vector<const tensor<T>*>& operands)
  {
    return detail::einsum_impl<T>(spec, operands);
  }

  /*!
   * \brief The matrix product of two tensors, as NumPy's `matmul`: each operand a stack of matrices along its leading
   *        axes, those axes broadcast against the other's (aligned from the right, an extent of 1 repeated), one product
   *        per matrix of the broadcast stack through the packed kernel, the products in parallel. A vector first is a
   *        row (its axis absent from the result), a vector second a column (likewise); two vectors give their dot
   *        product, of rank 0. The broadcast operands are not copied: a repeated matrix is read where it is.
   * \param[in] a,b The operands, of rank 1 or more.
   * \return `a @ b`, of the broadcast leading axes, then the rows of \p a and the columns of \p b.
   * \throws dimension_error If an operand is of rank 0, the inner extents differ, or the leading axes do not broadcast.
   */
  template <typename T>
  [[nodiscard]] tensor<T> matmul(const tensor<T>& a,
                                 const tensor<T>& b)
  {
    const std::vector<std::size_t>& sa {a.shape()};
    const std::vector<std::size_t>& sb {b.shape()};
    if (sa.empty() || sb.empty()) {
      throw dimension_error("matmul: a rank-0 operand has no matrix to multiply");
    }
    const bool        row    {sa.size() == 1};
    const bool        column {sb.size() == 1};
    const std::size_t m      {row ? 1 : sa[sa.size() - 2]};
    const std::size_t k      {sa.back()};
    const std::size_t n      {column ? 1 : sb.back()};
    if ((column ? sb[0] : sb[sb.size() - 2]) != k) {
      throw dimension_error("matmul: the inner extents of the operands differ");
    }
    const std::size_t        lead_a {sa.size() - (row ? 1 : 2)};
    const std::size_t        lead_b {sb.size() - (column ? 1 : 2)};
    const std::size_t        rank   {std::max(lead_a, lead_b)};
    std::vector<std::size_t> batch(rank);
    std::vector<std::size_t> stride_a(rank, 0); // 0 along an axis the operand repeats
    std::vector<std::size_t> stride_b(rank, 0);
    std::size_t              step_a {m * k};
    std::size_t              step_b {k * n};
    for (std::size_t d {rank}; d-- > 0;) {
      const std::size_t da {d + lead_a >= rank ? sa[d + lead_a - rank] : 1};
      const std::size_t db {d + lead_b >= rank ? sb[d + lead_b - rank] : 1};
      if (da != db && da != 1 && db != 1) {
        throw dimension_error("matmul: the leading axes of the operands do not broadcast");
      }
      batch[d]    = da == 1 ? db : da;
      stride_a[d] = da == 1 ? 0 : step_a;
      stride_b[d] = db == 1 ? 0 : step_b;
      step_a      = detail::checked_mul(step_a, da);
      step_b      = detail::checked_mul(step_b, db);
    }
    std::size_t count {1};
    for (const std::size_t extent : batch) {
      count = detail::checked_mul(count, extent);
    }
    std::vector<std::size_t> shape {batch};
    if (!row) {
      shape.push_back(m);
    }
    if (!column) {
      shape.push_back(n);
    }
    const std::array<std::size_t, 3> elements {count, m, n};
    buffer<T>                        c(detail::element_count(elements.begin(), elements.end()));
    if (rank == 0) { // no stack: one product
      detail::gemm(m, n, k, a.data().data(), k, b.data().data(), n, c.data(), n, detail::c_start::zero);
      return tensor<T> {std::move(shape), std::move(c)};
    }
    // B shared by the whole stack and A's matrices one after another: one product of all A's rows, B packed once
    // (1000 × 3 by 3 × 64 matrices times 64 × 64, ten AVX-512 cores, 165 → 87 µs). Each element of C the same.
    const bool shared_b {std::ranges::all_of(stride_b, [](std::size_t s) { return s == 0; }) && std::ranges::none_of(stride_a, [](std::size_t s) { return s == 0; })};
    if (shared_b) {
      detail::gemm(count * m, n, k, a.data().data(), k, b.data().data(), n, c.data(), n, detail::c_start::zero);
      return tensor<T> {std::move(shape), std::move(c)};
    }
    const auto product {[&](std::size_t p) {
                          std::size_t rest {p};
                          std::size_t offset_a {0};
                          std::size_t offset_b {0};
                          for (std::size_t d {batch.size()}; d-- > 0;) {
                            const std::size_t index {rest % batch[d]};
                            rest     /= batch[d];
                            offset_a += index * stride_a[d];
                            offset_b += index * stride_b[d];
                          }
                          detail::gemm(m, n, k, a.data().data() + offset_a, k, b.data().data() + offset_b, n, c.data() + (p * m * n), n, detail::c_start::zero);
                        }};
    // Fewer products than threads: one after another, each on all of them, where the stack's products in parallel held
    // each to one (two products of order 1000, ten AVX-512 cores: 28.0 → 7.2 ms); else the stack's products in parallel.
    if (count < max_threads()) {
      for (std::size_t p {0}; p < count; ++p) {
        product(p);
      }
      return tensor<T> {std::move(shape), std::move(c)};
    }
    detail::parallel_for(count, m * n * k / detail::gemm_kernel_speedup, product);
    return tensor<T> {std::move(shape), std::move(c)};
  }
} // namespace numerical

#endif // NUMERICAL_TENSOR_HPP
