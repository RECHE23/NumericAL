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
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "simd.hpp"
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
      const axis_walk walk                           {walk_along(t.shape(), axis)};
      buffer<T>       out(walk.outer * walk.inner);
      for (std::size_t o {0}; o < walk.outer; ++o) {
        for (std::size_t i {0}; i < walk.inner; ++i) {
          out[(o * walk.inner) + i] = reduce(t.data().data() + (o * walk.extent * walk.inner) + i, walk.extent, walk.inner);
        }
      }
      return tensor<T> {without_axis(t.shape(), axis), std::move(out)};
    }

    //! \brief `x · x`.
    template <typename T>
    [[nodiscard]] T square(const T& x)
    {
      return static_cast<T>(x * x);
    }

    //! \brief NumPy's complex square, (re·re − im·im) + (re·im + im·re)i, without the infinity handling of a
    //!        general product (C Annex G), which costs a call per element.
    template <typename U>
    [[nodiscard]] std::complex<U> square(const std::complex<U>& x)
    {
      return {(x.real() * x.real()) - (x.imag() * x.imag()), (x.real() * x.imag()) + (x.imag() * x.real())};
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

#if defined(__GNUC__)
    //! \brief The lanes where \p x is greater than \p best (all ones there, zeros elsewhere).
    template <typename V,
              typename T>
    [[nodiscard]] auto better(const V& x,
                              const V& best,
                              const std::greater<T>& /*larger*/)
    {
      return x > best;
    }

    //! \brief The lanes where \p x is less than \p best.
    template <typename V,
              typename T>
    [[nodiscard]] auto better(const V& x,
                              const V& best,
                              const std::less<T>& /*smaller*/)
    {
      return x < best;
    }

    //! \brief A lane differs from another only when unordered: \p a against itself flags the NaN lanes.
    template <typename V>
    [[nodiscard]] auto unordered(const V& a,
                                 const V& b)
    {
      return a != b;
    }

#endif

    /*!
     * \brief The largest (\p larger is `std::greater`) or smallest of \p count contiguous elements, or a NaN if
     *        one is among them, as NumPy's: the comparisons run without a branch over native vectors (or eight
     *        scalar lanes), each lane its own running best, and the NaN is tracked apart.
     */
    template <typename T,
              typename Order>
    [[nodiscard]] T contiguous_extreme(const T*     first,
                                       std::size_t  count,
                                       const Order& larger)
    {
      T              result   {first[0]};
      bool           nan      {false};
      std::size_t    done     {0};
#if defined(__GNUC__)
      using lane = typename native_vector<T>::type;
      constexpr std::size_t width {native_vector_bytes / sizeof(T)};
      constexpr std::size_t lanes {4}; // independent vectors, so no comparison waits on the one before it
      if (count >= width * lanes) {
        using mask = decltype(lane {} > lane {}); // all ones in a lane where a comparison holds
        std::array<lane, lanes> best {};
        std::array<mask, lanes> nans {};
        for (std::size_t l {0}; l < lanes; ++l) {
          std::memcpy(&best[l], first + (l * width), sizeof(lane));
        }
        done = count - (count % (width * lanes));
        for (std::size_t i {0}; i < done; i += width * lanes) {
#  pragma GCC unroll 4 // the lanes in registers: GCC keeps a rolled array of vectors in memory
          for (std::size_t l {0}; l < lanes; ++l) {
            lane x;
            std::memcpy(&x, first + i + (l * width), sizeof(lane));
            // Blended through the comparison's all-ones / all-zeros mask: both compilers keep it in vector
            // registers (GCC's `?:` on vectors does not).
            const mask take {better(x, best[l], larger)};
            best[l]  = std::bit_cast<lane>((std::bit_cast<mask>(x) & take) | (std::bit_cast<mask>(best[l]) & ~take));
            nans[l] |= unordered(x, x);
          }
        }
        for (std::size_t l {0}; l < lanes; ++l) {
          for (std::size_t k {0}; k < width; ++k) {
            nan    = nan || nans[l][k] != 0;
            result = larger(best[l][k], result) ? best[l][k] : result;
          }
        }
      }
#endif
      std::array<T, 8>        best {};
      std::array<unsigned, 8> nans {};
      best.fill(result);
      for (std::size_t i {done}; i < count; ++i) {
        const T x {first[i]};
        nans[i % 8] |= static_cast<unsigned>(std::isnan(x));
        best[i % 8]  = larger(x, best[i % 8]) ? x : best[i % 8];
      }
      for (std::size_t l {0}; l < 8; ++l) {
        nan    = nan || nans[l] != 0U;
        result = larger(best[l], result) ? best[l] : result;
      }
      return nan ? std::numeric_limits<T>::quiet_NaN() : result;
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
      if constexpr (std::is_floating_point_v<T>) {
        if (stride == 1) {
          // A contiguous line: the comparison without a branch (so it vectorises) and the NaN tracked apart,
          // a NaN anywhere is the answer, as NumPy's.
          return contiguous_extreme(first, count, larger);
        }
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

    /*!
     * \brief The largest (\p larger is `std::greater`) or smallest elements along \p axis. Along an inner axis
     *        each line is contiguous (\ref extreme); along an outer one the slabs are compared element by element,
     *        so the loop runs over contiguous memory rather than down a column.
     */
    template <typename T,
              typename Order>
    [[nodiscard]] tensor<T> extremes_along(const tensor<T>& t,
                                           std::size_t      axis,
                                           const Order&     larger)
    {
      const axis_walk walk {walk_along(t.shape(), axis)};
      // An empty axis has no maximum even when the result is empty too (NumPy refuses both).
      if (walk.extent == 0) {
        throw dimension_error("a zero-size reduction to a maximum or a minimum has no identity");
      }
      if (!std::is_floating_point_v<T> || walk.inner == 1) {
        return reduce_along(t, axis, [&larger](const T* first, std::size_t count, std::size_t stride) {
                              return extreme(first, count, stride, larger);
                            });
      }
      buffer<T>             out(walk.outer * walk.inner);
      std::vector<unsigned> nan(walk.inner);
      for (std::size_t o {0}; o < walk.outer; ++o) {
        const T* const slab {t.data().data() + (o * walk.extent * walk.inner)};
        T* const       best {out.data() + (o * walk.inner)};
        std::copy(slab, slab + walk.inner, best);
        std::fill(nan.begin(), nan.end(), 0U);
        for (std::size_t e {0}; e < walk.extent; ++e) {
          const T* const row {slab + (e * walk.inner)};
          for (std::size_t i {0}; i < walk.inner; ++i) {
            const T x {row[i]};
            nan[i]  |= static_cast<unsigned>(std::isnan(x));
            best[i]  = larger(x, best[i]) ? x : best[i];
          }
        }
        for (std::size_t i {0}; i < walk.inner; ++i) {
          if (nan[i] != 0U) {
            best[i] = std::numeric_limits<T>::quiet_NaN();
          }
        }
      }
      return tensor<T> {without_axis(t.shape(), axis), std::move(out)};
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
      buffer<T> out(a.size());
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
    buffer<T> out(count); // every element written below
    if (count == 0) {
      return tensor<T> {std::move(shape), std::move(out)};
    }
    // The innermost axis runs as one loop, each operand stepping 1 or 0 (a broadcast one) along it, so a
    // scalar or a row against a matrix is a plain vector loop; the odometer turns only between rows.
    const std::size_t        inner  {shape[rank - 1]};
    const std::size_t        step_a {sa[rank - 1]};
    const std::size_t        step_b {sb[rank - 1]};
    std::vector<std::size_t> index(rank, 0);
    std::size_t              ia     {0};
    std::size_t              ib     {0};
    for (std::size_t k {0}; k < count; k += inner) {
      const T* const pa {a.data().data() + ia};
      const T* const pb {b.data().data() + ib};
      T* const       po {out.data() + k};
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
    buffer<T> out(a.size());
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
    buffer<R> out(t.size());
    std::transform(t.data().begin(), t.data().end(), out.begin(), f);
    return tensor<R> {t.shape(), std::move(out)};
  }

  /*!
   * \brief `a` to the power `b` element by element, broadcast, as `std::pow`; a single exponent of 2, 0.5 or
   *        -1 is computed as NumPy computes it, `x·x`, `sqrt(x)` or `1/x` (exact where `pow` rounds: a complex
   *        square has no stray real part).
   */
  template <typename T>
  [[nodiscard]] tensor<T> pow(const tensor<T>& a,
                              const tensor<T>& b)
  {
    if constexpr (!std::is_integral_v<T>) {
      if (b.size() == 1 && b.shape().size() <= a.shape().size()) {
        const T y {b.data()[0]};
        if (y == T(2)) {
          return map(a, [](const T& x) { return detail::square(x); });
        }
        if (y == T(0.5)) {
          return map(a, [](const T& x) { return static_cast<T>(std::sqrt(x)); });
        }
        if (y == T(-1)) {
          return map(a, [](const T& x) { return static_cast<T>(T(1) / x); });
        }
      }
    }
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
    const detail::axis_walk walk {detail::walk_along(t.shape(), axis)};
    buffer<T>               out(walk.outer * walk.inner);
    detail::pairwise_sums(t.data().data(), walk.outer, walk.extent, walk.inner, out.data());
    return tensor<T> {detail::without_axis(t.shape(), axis), std::move(out)};
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
    const detail::axis_walk walk  {detail::walk_along(t.shape(), axis)};
    const auto              count {static_cast<T>(static_cast<double>(walk.extent))};
    buffer<T>               out(walk.outer * walk.inner);
    detail::pairwise_sums(t.data().data(), walk.outer, walk.extent, walk.inner, out.data());
    for (T& x : out) {
      x = x / count;
    }
    return tensor<T> {detail::without_axis(t.shape(), axis), std::move(out)};
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
    return detail::extremes_along(t, axis, std::greater<T> {});
  }

  //! \brief The smallest elements along \p axis. \throws dimension_error If \p axis is not an axis or is empty.
  template <std::totally_ordered T>
  [[nodiscard]] tensor<T> min(const tensor<T>& t,
                              std::size_t      axis)
  {
    return detail::extremes_along(t, axis, std::less<T> {});
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
    return tensor<T> {std::move(shape), buffer<T>(count, value)};
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
    buffer<T> out(detail::checked_mul(n, n), T {});
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
    const double        length            {std::ceil((stop - start) / step)};
    const std::size_t   count             {length > 0.0 ? static_cast<std::size_t>(length) : 0};
    buffer<double>      out(count);
    const double        delta             {(start + step) - start};
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
    buffer<double>      out(count);
    const double        step              {count > 1 ? (stop - start) / static_cast<double>(count - 1) : 0.0};
    for (std::size_t i {0}; i < count; ++i) {
      // The product rounded before the sum, as NumPy's linspace computes it in two operations. A compiler
      // may fuse `i * step` into the addition after it (GCC does, across statements, wherever there is a
      // fused multiply-add); a product computed by fma, `i * step - 0`, is exactly the rounded one, signed
      // zero included, and no multiplication is left to fuse.
      const double offset {std::fma(static_cast<double>(i), step, -0.0)};
      out[i] = start + offset;
    }
    if (count > 1) {
      out[count - 1] = stop;
    }
    return tensor<double> {{count}, std::move(out)};
  }
} // namespace numerical

#endif // NUMERICAL_TENSOR_OPS_HPP
