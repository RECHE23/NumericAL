/*!
 * \file tensor_ops.hpp
 * \brief Element-wise arithmetic with NumPy's broadcasting, reductions, reshaping and creation of tensors.
 *
 * Two shapes broadcast as NumPy's do: aligned on their last axes, each pair of extents equal or one of them
 * 1 (a missing leading axis counts as 1), the result taking the larger. A sum adds by pairwise halving, as
 * NumPy's reference algorithm does (NumPy's vector builds group the terms their own way, so the last bits
 * can differ by platform): its rounding error grows with the logarithm of the count instead of the count. A maximum or
 * a minimum is defined for an ordered scalar only, and an empty reduction without an identity throws.
 */
#ifndef NUMERICAL_TENSOR_OPS_HPP
#define NUMERICAL_TENSOR_OPS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensor.hpp"

namespace numerical {

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

    /*!
     * \brief Sum of `count` values from \p first at stride \p stride, by NumPy's pairwise summation: below 8
     *        values in order; up to 128 in eight interleaved partial sums, added pairwise, then the
     *        remainder in order; above, the two halves (the first a multiple of 8 long) summed apart.
     */
    template <typename T>
    [[nodiscard]] T pairwise_sum(const T*    first,
                                 std::size_t count,
                                 std::size_t stride)
    {
      constexpr std::size_t lanes {8};
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

    //! \brief The extents and strides that walk a tensor of \p shape with axis \p axis taken out: `outer`
    //!        positions before it, `inner` after it, the axis itself `extent` long at stride `inner`.
    struct axis_walk
    {
      std::size_t outer;  //!< The product of the extents before the axis.
      std::size_t extent; //!< The axis' extent.
      std::size_t inner;  //!< The product of the extents after the axis.
    };

    //! \brief \ref axis_walk of \p shape along \p axis. \throws dimension_error If \p axis is not an axis.
    [[nodiscard]] inline axis_walk walk_along(const std::vector<std::size_t>& shape,
                                              std::size_t                     axis)
    {
      if (axis >= shape.size()) {
        throw dimension_error("axis " + std::to_string(axis) + " is out of bounds for a tensor of rank "
                              + std::to_string(shape.size()));
      }
      axis_walk walk {.outer = 1, .extent = shape[axis], .inner = 1};
      for (std::size_t i {0}; i < axis; ++i) {
        walk.outer *= shape[i];
      }
      for (std::size_t i {axis + 1}; i < shape.size(); ++i) {
        walk.inner *= shape[i];
      }
      return walk;
    }

    //! \brief The shape with axis \p axis removed.
    [[nodiscard]] inline std::vector<std::size_t> without_axis(std::vector<std::size_t> shape,
                                                               std::size_t              axis)
    {
      shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(axis));
      return shape;
    }

    /*!
     * \brief Reduces each line of \p t along \p axis by \p reduce, which takes the line's first element,
     *        its length and its stride.
     */
    template <typename T,
              typename Reduce>
    [[nodiscard]] tensor<T> reduce_along(const tensor<T>& t,
                                         std::size_t      axis,
                                         const Reduce&    reduce)
    {
      const axis_walk walk {walk_along(t.shape(), axis)};
      std::vector<T>  out(walk.outer * walk.inner);
      for (std::size_t o {0}; o < walk.outer; ++o) {
        for (std::size_t i {0}; i < walk.inner; ++i) {
          out[(o * walk.inner) + i] = reduce(t.data().data() + (o * walk.extent * walk.inner) + i, walk.extent, walk.inner);
        }
      }
      return tensor<T> {without_axis(t.shape(), axis), std::move(out)};
    }

    //! \brief A NaN? Never for a scalar without one.
    template <typename T>
    [[nodiscard]] bool is_nan(const T& /*x*/)
    {
      return false;
    }

    //! \brief A NaN?
    [[nodiscard]] inline bool is_nan(double x)
    {
      return std::isnan(x);
    }

    //! \brief The largest (\p larger is `std::greater`) or smallest element of a line.
    template <typename T,
              typename Order>
    [[nodiscard]] T extreme(const T*     first,
                            std::size_t  count,
                            std::size_t  stride,
                            const Order& larger)
    {
      if (count == 0) {
        throw dimension_error("a zero-size reduction to a maximum or a minimum has no identity");
      }
      T best {first[0]};
      for (std::size_t i {1}; i < count; ++i) {
        // A NaN wins, as in NumPy: every comparison with it is false, so a NaN candidate is taken apart, and
        // once taken nothing compares larger than it.
        const T& candidate {first[i * stride]};
        if (is_nan(candidate) || larger(candidate, best)) {
          best = candidate;
        }
      }
      return best;
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

  /*!
   * \brief `op(a, b)` element by element, the operands broadcast against each other.
   * \throws dimension_error If the shapes do not broadcast.
   */
  template <typename T,
            typename Op>
  [[nodiscard]] tensor<T> broadcast_apply(const tensor<T>& a,
                                          const tensor<T>& b,
                                          const Op&        op)
  {
    if (a.shape() == b.shape()) {
      std::vector<T> out(a.size());
      std::transform(a.data().begin(), a.data().end(), b.data().begin(), out.begin(), op);
      return tensor<T> {a.shape(), std::move(out)};
    }
    std::vector<std::size_t>       shape {broadcast_shape(a.shape(), b.shape())};
    const std::size_t              rank  {shape.size()};
    const std::vector<std::size_t> sa    {detail::broadcast_strides(a.shape(), rank)};
    const std::vector<std::size_t> sb    {detail::broadcast_strides(b.shape(), rank)};
    std::size_t                    count {1};
    for (const std::size_t extent : shape) {
      count *= extent;
    }
    std::vector<T>           out(count);
    std::vector<std::size_t> index(rank, 0);
    std::size_t              ia {0};
    std::size_t              ib {0};
    for (std::size_t k {0}; k < count; ++k) {
      out[k] = op(a.data()[ia], b.data()[ib]);
      for (std::size_t axis {rank}; axis-- > 0;) { // odometer: carry into the next axis on wrap
        ++index[axis];
        ia += sa[axis];
        ib += sb[axis];
        if (index[axis] < shape[axis]) {
          break;
        }
        ia          -= sa[axis] * shape[axis];
        ib          -= sb[axis] * shape[axis];
        index[axis]  = 0;
      }
    }
    return tensor<T> {std::move(shape), std::move(out)};
  }

  //! \brief Element-wise sum, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator+(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::plus<T> {});
  }

  //! \brief Element-wise difference, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator-(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::minus<T> {});
  }

  //! \brief Element-wise product, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator*(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::multiplies<T> {});
  }

  //! \brief Element-wise quotient, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator/(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::divides<T> {});
  }

  //! \brief Element-wise negation.
  template <typename T>
  [[nodiscard]] tensor<T> operator-(const tensor<T>& a)
  {
    std::vector<T> out(a.size());
    std::transform(a.data().begin(), a.data().end(), out.begin(), std::negate<T> {});
    return tensor<T> {a.shape(), std::move(out)};
  }

  //! \brief `f(x)` for every element, in a tensor of the same shape and of `f`'s result type.
  template <typename T,
            typename F>
  [[nodiscard]] auto map(const tensor<T>& t,
                         const F&         f) -> tensor<std::decay_t<decltype(f(std::declval<const T&>()))>>
  {
    using R = std::decay_t<decltype(f(std::declval<const T&>()))>;
    std::vector<R> out(t.size());
    std::transform(t.data().begin(), t.data().end(), out.begin(), f);
    return tensor<R> {t.shape(), std::move(out)};
  }

  //! \brief `a` to the power `b` element by element, broadcast, as `std::pow`.
  template <typename T>
  [[nodiscard]] tensor<T> pow(const tensor<T>& a,
                              const tensor<T>& b)
  {
    return broadcast_apply(a, b, [](const T& x, const T& y) { return static_cast<T>(std::pow(x, y)); });
  }

  //! \brief The sum of every element, by pairwise halving (0 for an empty tensor).
  template <typename T>
  [[nodiscard]] T sum(const tensor<T>& t)
  {
    return detail::pairwise_sum(t.data().data(), t.size(), 1);
  }

  //! \brief The sums along \p axis, by pairwise halving. \throws dimension_error If \p axis is not an axis.
  template <typename T>
  [[nodiscard]] tensor<T> sum(const tensor<T>& t,
                              std::size_t      axis)
  {
    return detail::reduce_along(t, axis, [](const T* first, std::size_t count, std::size_t stride) {
                                  return detail::pairwise_sum(first, count, stride);
                                });
  }

  /*!
   * \brief The mean of every element. An empty tensor's mean is NaN, as NumPy's (which also warns).
   */
  template <typename T>
  [[nodiscard]] T mean(const tensor<T>& t)
  {
    return sum(t) / static_cast<T>(static_cast<double>(t.size()));
  }

  //! \brief The means along \p axis. \throws dimension_error If \p axis is not an axis.
  template <typename T>
  [[nodiscard]] tensor<T> mean(const tensor<T>& t,
                               std::size_t      axis)
  {
    return detail::reduce_along(t, axis, [](const T* first, std::size_t count, std::size_t stride) {
                                  return detail::pairwise_sum(first, count, stride) / static_cast<T>(static_cast<double>(count));
                                });
  }

  //! \brief The largest element (a NaN, if there is one). \throws dimension_error If \p t is empty.
  template <std::totally_ordered T>
  [[nodiscard]] T max(const tensor<T>& t)
  {
    return detail::extreme(t.data().data(), t.size(), 1, std::greater<T> {});
  }

  //! \brief The smallest element (a NaN, if there is one). \throws dimension_error If \p t is empty.
  template <std::totally_ordered T>
  [[nodiscard]] T min(const tensor<T>& t)
  {
    return detail::extreme(t.data().data(), t.size(), 1, std::less<T> {});
  }

  //! \brief The largest elements along \p axis. \throws dimension_error If \p axis is not an axis or is empty.
  template <std::totally_ordered T>
  [[nodiscard]] tensor<T> max(const tensor<T>& t,
                              std::size_t      axis)
  {
    return detail::reduce_along(t, axis, [](const T* first, std::size_t count, std::size_t stride) {
                                  return detail::extreme(first, count, stride, std::greater<T> {});
                                });
  }

  //! \brief The smallest elements along \p axis. \throws dimension_error If \p axis is not an axis or is empty.
  template <std::totally_ordered T>
  [[nodiscard]] tensor<T> min(const tensor<T>& t,
                              std::size_t      axis)
  {
    return detail::reduce_along(t, axis, [](const T* first, std::size_t count, std::size_t stride) {
                                  return detail::extreme(first, count, stride, std::less<T> {});
                                });
  }

  /*!
   * \brief The same elements in the same order under another shape.
   * \throws dimension_error If the shapes hold different counts of elements.
   */
  template <typename T>
  [[nodiscard]] tensor<T> reshape(const tensor<T>&         t,
                                  std::vector<std::size_t> shape)
  {
    std::size_t count {1};
    for (const std::size_t extent : shape) {
      count = detail::checked_mul(count, extent);
    }
    if (count != t.size()) {
      throw dimension_error("cannot reshape a tensor of size " + std::to_string(t.size()) + " into shape "
                            + detail::shape_text(shape));
    }
    return tensor<T> {std::move(shape), t.data()};
  }

  //! \brief A tensor of \p shape with every element \p value.
  template <typename T>
  [[nodiscard]] tensor<T> full(std::vector<std::size_t> shape,
                               const T&                 value)
  {
    std::size_t count {1};
    for (const std::size_t extent : shape) {
      count = detail::checked_mul(count, extent);
    }
    return tensor<T> {std::move(shape), std::vector<T>(count, value)};
  }

  //! \brief A tensor of \p shape full of zeros.
  template <typename T>
  [[nodiscard]] tensor<T> zeros(std::vector<std::size_t> shape)
  {
    return full<T>(std::move(shape), T {});
  }

  //! \brief A tensor of \p shape full of ones.
  template <typename T>
  [[nodiscard]] tensor<T> ones(std::vector<std::size_t> shape)
  {
    return full<T>(std::move(shape), static_cast<T>(1));
  }

  //! \brief The `n × n` identity.
  template <typename T>
  [[nodiscard]] tensor<T> eye(std::size_t n)
  {
    std::vector<T> out(detail::checked_mul(n, n), T {});
    for (std::size_t i {0}; i < n; ++i) {
      out[(i * n) + i] = static_cast<T>(1);
    }
    return tensor<T> {{n, n}, std::move(out)};
  }

  /*!
   * \brief `start, start + step, …` below \p stop (above it for a negative step), as NumPy's `arange`: its
   *        length is `ceil((stop − start) / step)` and element `i` is `start + i·delta` with
   *        `delta = (start + step) − start`, the step as the first two elements hold it,
   *        rounded once (a fused multiply-add, as NumPy's arm64 builds compute it; one without fusion rounds
   *        the product first and can differ in the last bits).
   * \throws dimension_error If \p step is zero or not finite.
   */
  [[nodiscard]] inline tensor<double> arange(double start,
                                             double stop,
                                             double step)
  {
    if (step == 0.0 || !std::isfinite(step) || !std::isfinite(start) || !std::isfinite(stop)) {
      throw dimension_error("arange needs finite bounds and a finite, non-zero step");
    }
    const double        length {std::ceil((stop - start) / step)};
    const std::size_t   count  {length > 0.0 ? static_cast<std::size_t>(length) : 0};
    std::vector<double> out(count);
    const double        delta  {(start + step) - start};
    for (std::size_t i {0}; i < count; ++i) {
      out[i] = std::fma(static_cast<double>(i), delta, start);
    }
    return tensor<double> {{count}, std::move(out)};
  }

  /*!
   * \brief \p count values evenly spaced from \p start to \p stop, both included, as NumPy's `linspace`:
   *        element `i` is `start + i·step` with `step = (stop − start) / (count − 1)`, and the last is
   *        \p stop exactly.
   */
  [[nodiscard]] inline tensor<double> linspace(double      start,
                                               double      stop,
                                               std::size_t count)
  {
    std::vector<double> out(count);
    const double        step {count > 1 ? (stop - start) / static_cast<double>(count - 1) : 0.0};
    for (std::size_t i {0}; i < count; ++i) {
      const double offset {static_cast<double>(i) * step}; // rounded before the sum, as NumPy's linspace
      out[i] = start + offset;
    }
    if (count > 1) {
      out[count - 1] = stop;
    }
    return tensor<double> {{count}, std::move(out)};
  }
} // namespace numerical

#endif // NUMERICAL_TENSOR_OPS_HPP
