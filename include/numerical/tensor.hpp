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
#include <cctype>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
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
           std::vector<T>           elements)
      : shape_(std::move(shape)), data_(std::move(elements))
    {
      finalize();
    }

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
    [[nodiscard]] const std::vector<T>& data() const noexcept
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
    std::vector<T>           data_;    //!< The elements, row-major.
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
    [[nodiscard]] tensor<T> tile_gemm(const std::vector<T>&    a,
                                      const std::vector<T>&    b,
                                      std::size_t              rows,
                                      std::size_t              inner,
                                      std::size_t              cols,
                                      std::vector<std::size_t> out_shape)
    {
      std::vector<T> c(rows * cols, T {}); // the caller routes only an output of two elements or more here
      if (inner != 0) {
        gemm(rows, cols, inner, a.data(), inner, b.data(), cols, c.data(), cols);
      }
      return tensor<T> {std::move(out_shape), std::move(c)};
    }

    //! \brief The columns a thread takes of a vector times a matrix: each of its rows added to them in order.
    inline constexpr std::size_t vector_chunk {128};

    /*!
     * \brief The elements of \p data, whose axes carry the labels \p from (extents in \p sizes), laid out row-major
     *        along the labels \p to, a permutation of \p from that differs from it (so at least one label): an
     *        odometer over \p to's outer labels, a strided sweep of its last.
     */
    template <typename T>
    [[nodiscard]] std::vector<T> permuted(const std::vector<T>&              data,
                                          const std::string&                 from,
                                          const std::string&                 to,
                                          const std::map<char, std::size_t>& sizes)
    {
      std::vector<T> out(data.size());
      if (data.empty()) {
        return out;
      }
      if (to.size() == 2) { // the one permutation of two axes, a transpose: by tiles, each read and written in the cache
        const std::size_t rows {sizes.at(from[0])};
        const std::size_t cols {sizes.at(from[1])};
        parallel_for((rows + transpose_tile - 1) / transpose_tile, transpose_tile * cols, [&](std::size_t block) {
                       const std::size_t r0 {block * transpose_tile};
                       const std::size_t r1 {std::min(r0 + transpose_tile, rows)};
                       for (std::size_t c0 {0}; c0 < cols; c0 += transpose_tile) {
                         const std::size_t c1 {std::min(c0 + transpose_tile, cols)};
                         for (std::size_t r {r0}; r < r1; ++r) {
                           for (std::size_t c {c0}; c < c1; ++c) {
                             out[(c * rows) + r] = data[(r * cols) + c];
                           }
                         }
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
      const std::vector<T> left                                               {sa == left_order ? std::vector<T> {} : permuted(a.data(), sa, left_order, sizes)};
      const std::vector<T> right                                              {sb == right_order ? std::vector<T> {} : permuted(b.data(), sb, right_order, sizes)};
      const T* const       left_data                                          {sa == left_order ? a.data().data() : left.data()};
      const T* const       right_data                                         {sb == right_order ? b.data().data() : right.data()};
      std::vector<T>       c(checked_mul(batches, checked_mul(rows, cols)), T {});
      if (rows == 1 && cols == 1) { // a dot product per batch, as `chunked_dot` sums each chunk
        parallel_for(batches, inner, [&](std::size_t p) {
                       c[p] = dot_product(left_data + (p * inner), right_data + (p * inner), inner);
                     });
      }
      else {
        parallel_for(batches, rows * cols * inner / gemm_kernel_speedup, [&](std::size_t p) {
                       gemm(rows, cols, inner, left_data + (p * rows * inner), inner, right_data + (p * inner * cols), cols,
                            c.data() + (p * rows * cols), cols);
                     });
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
    template <typename T>
    [[nodiscard]] tensor<T> einsum_impl(const std::string&                    spec_in,
                                        const std::vector<const tensor<T>*>&  operands)
    {
      // Tolerate whitespace anywhere in the spec, the way NumPy does ("ij, jk -> ik").
      std::string spec {spec_in};
      spec.erase(std::remove_if(spec.begin(), spec.end(),
                                [](unsigned char ch) { return std::isspace(ch) != 0; }),
                 spec.end());
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
          std::size_t              rows  {1};
          std::size_t              cols  {1};
          std::vector<std::size_t> out_shape;
          for (std::size_t k {0}; k + 1 < s0.size(); ++k) {
            rows *= sizes.at(s0[k]);
            out_shape.push_back(sizes.at(s0[k]));
          }
          for (std::size_t k {1}; k < s1.size(); ++k) {
            cols *= sizes.at(s1[k]);
            out_shape.push_back(sizes.at(s1[k]));
          }
          if (cols == 1 && rows > 1 && inner != 0) { // a matrix times a vector: each row's dot product, as "i,i->"
            const T* const matrix {operands[0]->data().data()};
            const T* const vector {operands[1]->data().data()};
            std::vector<T> y(rows);
            matrix_times_vector(matrix, rows, inner, vector, y.data());
            return tensor<T> {std::move(out_shape), std::move(y)};
          }
          if (rows == 1 && cols > 1 && inner != 0) { // a vector times a matrix: its rows scaled and added in order
            const T* const vector    {operands[0]->data().data()};
            const T* const matrix    {operands[1]->data().data()};
            std::vector<T> y(cols, T {});
            parallel_for((cols + vector_chunk - 1) / vector_chunk, inner * vector_chunk, [&](std::size_t chunk) {
                           const std::size_t first {chunk * vector_chunk};
                           const std::size_t count {std::min(vector_chunk, cols - first)};
                           for (std::size_t k {0}; k < inner; ++k) {
                             add_product(y.data() + first, vector[k], matrix + (k * cols) + first, count);
                           }
                         });
            return tensor<T> {std::move(out_shape), std::move(y)};
          }
          if (rows * cols != 1) { // the product, an empty one included; a single element is one dot product
            return tile_gemm<T>(operands[0]->data(), operands[1]->data(),
                                rows, inner, cols, std::move(out_shape));
          }
          return tensor<T> {std::move(out_shape), std::vector<T> {chunked_dot(operands[0]->data().data(), operands[1]->data().data(), inner)}};
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
      std::size_t out_outer {1};
      for (std::size_t k {0}; k < free_outer_end; ++k) {
        out_outer *= extent[k];
      }
      std::size_t sum_outer {1};
      for (std::size_t k {n_free}; k < sum_outer_end; ++k) {
        sum_outer = detail::checked_mul(sum_outer, extent[k]); // a wrapped count would sum nothing, silently
      }
      std::vector<std::size_t> out_shape(extent.begin(),
                                         extent.begin() + static_cast<std::ptrdiff_t>(n_free));

      std::vector<std::size_t> index(n_labels, 0); // value of each outer label
      std::vector<std::size_t> offset(n_ops, 0);   // each operand's flat offset (outer labels only)
      std::vector<T>           result;
      result.reserve(out_size);

      for (std::size_t of {0}; of < out_outer; ++of) {
        for (std::size_t fe {0}; fe < inner_free; ++fe) {
          T acc {};
          for (std::size_t os {0}; os < sum_outer; ++os) {
            for (std::size_t se {0}; se < inner_sum; ++se) {
              T prod {1};
              for (std::size_t i {0}; i < n_ops; ++i) {
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
} // namespace numerical

#endif // NUMERICAL_TENSOR_HPP
