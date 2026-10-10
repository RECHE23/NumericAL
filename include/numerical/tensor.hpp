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
 * ascending), so results are reproducible.
 */
#ifndef NUMERICAL_TENSOR_HPP
#define NUMERICAL_TENSOR_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

    //! \brief A rank-2 tensor from a matrix. \param[in] m The matrix.
    explicit tensor(const matrix<T>& m)
      : shape_ {m.rows(), m.cols()}
    {
      data_.reserve(m.rows() * m.cols());
      for (std::size_t i {0}; i < m.rows(); ++i) {
        for (std::size_t j {0}; j < m.cols(); ++j) {
          data_.push_back(m(i, j));
        }
      }
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
     * \brief Cache-blocked matrix product `C(M×N) = A(M×K) · B(K×N)`, row-major.
     *
     * The fast path for a plain two-operand single-contraction einsum (matmul,
     * matrix–vector, dot). Iterating in m-k-n order accumulating into `C` keeps
     * every access unit-stride and reads each operand block while it is cache
     * resident; the contraction is summed in ascending order, so the result is
     * bit-identical to the general engine (no tolerance change). The innermost
     * `n` loop is a contiguous multiply-accumulate the compiler auto-vectorises.
     */
    template <typename T>
    [[nodiscard]] tensor<T> tile_gemm(const std::vector<T>&    a,
                                      const std::vector<T>&    b,
                                      std::size_t              rows,
                                      std::size_t              inner,
                                      std::size_t              cols,
                                      std::vector<std::size_t> out_shape)
    {
      constexpr std::size_t tile             {32};
      std::vector<T>        c(rows * cols, T {});
      // Each row-block owns disjoint rows of C, so the blocks run in parallel
      // without locks and the result is identical to a sequential run.
      const std::size_t row_blocks {((rows + tile) - 1) / tile};
      parallel_for(row_blocks, tile * inner * cols / vector_speedup, [&](std::size_t block) {
                     const std::size_t mm {block * tile};
                     const std::size_t m_end {std::min(mm + tile, rows)};
                     for (std::size_t kk {0}; kk < inner; kk += tile) {
                       for (std::size_t nn {0}; nn < cols; nn += tile) {
                         const std::size_t k_end {std::min(kk + tile, inner)};
                         const std::size_t n_end {std::min(nn + tile, cols)};
                         for (std::size_t m {mm}; m < m_end; ++m) {
                           for (std::size_t k {kk}; k < k_end; ++k) {
                             const T scale {a[(m * inner) + k]};
                             for (std::size_t n {nn}; n < n_end; ++n) {
                               c[(m * cols) + n] += scale * b[(k * cols) + n];
                             }
                           }
                         }
                       }
                     }
                   });
      return tensor<T> {std::move(out_shape), std::move(c)};
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
      const std::size_t arrow {spec.find("->")};
      if (arrow == std::string::npos) {
        throw einsum_error("einsum: spec must contain '->'");
      }
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
      // carries, then contract them two at a time, left to right, each intermediate keeping only the
      // labels still to come. One loop over every label of `ij,jk,kl->il` runs n⁴ terms; this runs 2·n³,
      // and `i,j,k,l,m->` over 65 536 ones each runs five sums of 65 536 instead of 2⁸⁰ terms.
      if (operands.size() > 2) {
        std::vector<tensor<T>>   reduced;
        std::vector<std::string> kept;
        reduced.reserve(operands.size());
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
          const std::vector<const tensor<T>*> alone {operands[i]};
          reduced.push_back(einsum_impl<T>(inputs[i] + "->" + keep, alone));
          kept.push_back(std::move(keep));
        }
        tensor<T>   accumulated {std::move(reduced.front())};
        std::string labels      {kept.front()};
        for (std::size_t i {1}; i < reduced.size(); ++i) {
          std::string later {output};
          for (std::size_t j {i + 1}; j < kept.size(); ++j) {
            later += kept[j];
          }
          std::string keep;
          for (const char label : labels + kept[i]) {
            if (later.find(label) != std::string::npos && keep.find(label) == std::string::npos) {
              keep.push_back(label);
            }
          }
          const std::vector<const tensor<T>*> pair {&accumulated, &reduced[i]};
          tensor<T>                           next {einsum_impl<T>(labels + "," + kept[i] + "->" + (i + 1 == reduced.size() ? output : keep), pair)};
          accumulated = std::move(next);
          labels      = std::move(keep);
        }
        return accumulated;
      }

      // Fast path: a plain two-operand single contraction in canonical layout —
      // `A` ending with the contracted label, `B` starting with it, and the
      // output being A's leading free labels then B's trailing ones — is exactly
      // a matrix product, routed to the cache-blocked kernel. (Any stray copy of
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
          // Tiling pays off only when there is an output to amortise it over; a
          // 1×1 result (a plain dot) is left to the general engine's tight loop.
          if (rows * cols > 1) {
            return tile_gemm<T>(operands[0]->data(), operands[1]->data(),
                                rows, inner, cols, std::move(out_shape));
          }
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
