/*!
 * \file simd.hpp
 * \brief The native vector of `float` and `double` the kernels hold their lanes in (GNU vector extensions,
 *        under GCC and clang; a kernel keeps a scalar form beside it for other compilers).
 */
#ifndef NUMERICAL_SIMD_HPP
#define NUMERICAL_SIMD_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <type_traits>

// A function inlined wherever it is called, past the compiler's budget: a call there costs more than its body.
#if defined(_MSC_VER) && !defined(__clang__)
#  define NUMERICAL_ALWAYS_INLINE __forceinline
#else
#  define NUMERICAL_ALWAYS_INLINE [[gnu::always_inline]] inline // NOLINT(cppcoreguidelines-macro-usage): an attribute, which no function can stand for
#endif

namespace numerical::detail {

  //! \brief The side of the tiles a transposing copy goes by (an `einsum` permutation of two axes, the eigenvectors'
  //!        columns): a tile read and a tile written fit the L1 cache, where element by element each write, a row
  //!        apart, missed it (order 1024 on arm64: 4.7 → 1.5 ms; the eigenvectors of order 1024, a tenth of an `eigh`
  //!        on six x86-64 cores).
  inline constexpr std::size_t transpose_tile {32};

  /*!
   * \brief The side of the square pieces a transposing copy walks a tile by, from the bytes between rows of its source
   *        and of its result: a tile's 32 rows a multiple of a power of two of bytes apart fall in a few of the L1
   *        cache's sets and evict one another, where 8 rows fit them. Pieces of 8 where either stride is a multiple of
   *        1 KiB on x86-64 (sets every 4 KiB: four AVX2 cores, 256 × 256 doubles 165 → 51 µs, 128 × 128 15.8 →
   *        10.3, 2048 × 2048 15.4 → 9.2 ms; ten AVX-512 cores 256 × 256 142 → 51 µs), of 8 KiB on arm64 (M1 Pro,
   *        1024 × 1024 3.8 → 0.95 ms, 2048 × 2048 19.7 → 7.2); elsewhere the whole tile, which pieces slowed (100 ×
   *        100 on the AVX2 cores 10.5 → 11.6 µs, most orders on the M1 Pro 10 to 40 %).
   */
  [[nodiscard]] inline std::size_t transpose_piece(std::size_t from_bytes,
                                                   std::size_t to_bytes) noexcept
  {
#if defined(__x86_64__) || defined(_M_X64)
    constexpr std::size_t aliased {std::size_t {1} << 10U};
#else
    constexpr std::size_t aliased {std::size_t {1} << 13U};
#endif
    return from_bytes % aliased == 0 || to_bytes % aliased == 0 ? 8 : transpose_tile;
  }

  /*!
   * \brief `copy(i, j)` for every `i` in `[i0, i1)` and `j` in `[j0, j1)`, by square pieces of side \p Piece, `j` the
   *        faster within a piece.
   */
  template <std::size_t Piece,
            typename Copy>
  void transpose_pieces(std::size_t i0,
                        std::size_t i1,
                        std::size_t j0,
                        std::size_t j1,
                        const Copy& copy)
  {
    for (std::size_t pi {i0}; pi < i1; pi += Piece) {
      const std::size_t pi1 {std::min(pi + Piece, i1)};
      for (std::size_t pj {j0}; pj < j1; pj += Piece) {
        const std::size_t pj1 {std::min(pj + Piece, j1)};
        for (std::size_t i {pi}; i < pi1; ++i) {
          for (std::size_t j {pj}; j < pj1; ++j) {
            copy(i, j);
          }
        }
      }
    }
  }

  /*!
   * \brief \ref transpose_pieces of side \p piece (\ref transpose_piece's 8 or the whole tile), each side its own
   *        instance: a side known only at run time slowed the whole tile's copy (100 × 100 doubles, four AVX2 cores,
   *        10.4 → 11.2 µs).
   */
  template <typename Copy>
  void transpose_walk(std::size_t i0,
                      std::size_t i1,
                      std::size_t j0,
                      std::size_t j1,
                      std::size_t piece,
                      const Copy& copy)
  {
    if (piece == 8) {
      transpose_pieces<8>(i0, i1, j0, j1, copy);
    }
    else {
      transpose_pieces<transpose_tile>(i0, i1, j0, j1, copy);
    }
  }

  //! \brief The rows of a lower triangle \ref symmetric_rows takes at once.
  inline constexpr std::size_t symmetric_row_group {4};

  /*!
   * \brief The columns `done…i+r` of \ref symmetric_rows, one by one: for each row `r` in order, `out[k] += row_r[k]
   *        · v[i+r]` for `done ≤ k < i + r` and `row_r[k] · v[k]` added to `sums[r]`, its diagonal term last.
   */
  template <typename T>
  [[nodiscard]] std::array<T, symmetric_row_group> symmetric_rows_tail(const T *                          rows,
                                                                       std::size_t                        n,
                                                                       const T *                          v,
                                                                       T *                                out,
                                                                       std::size_t                        i,
                                                                       std::size_t                        done,
                                                                       std::array<T, symmetric_row_group> sums)
  {
    for (std::size_t r {0}; r < symmetric_row_group; ++r) {
      const T*  row    {rows + (r * n)};
      const T   factor {v[i + r]};
      for (std::size_t k {done}; k < i + r; ++k) {
        sums[r] += row[k] * v[k];
        out[k]  += row[k] * factor;
      }
      sums[r] += row[i + r] * factor;
    }
    return sums;
  }

  //! \brief A complex number of `float` or `double`, which `gemm_split` multiplies through the real kernel and the
  //!        kernels take apart into its real and imaginary parts.
  template <typename T>
  concept split_complex = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

#if defined(__GNUC__)
  //! \brief The native vector width, in bytes.
#  if defined(__AVX__)
  inline constexpr std::size_t native_vector_bytes {32};
#  else
  inline constexpr std::size_t native_vector_bytes {16};
#  endif

  //! \brief A native vector of `T`, for the types the kernels hold in vectors.
  template <typename T>
  struct native_vector;

  //! \brief A native vector of `float`.
  template <>
  struct native_vector<float>
  {
    using type = float __attribute__((vector_size(native_vector_bytes))); //!< The vector.
  };

  //! \brief A native vector of `double`.
  template <>
  struct native_vector<double>
  {
    using type = double __attribute__((vector_size(native_vector_bytes))); //!< The vector.
  };

  //! \brief Whether `T` is held in \ref native_vector.
  template <typename T>
  concept native_scalar = std::is_same_v<T, float> || std::is_same_v<T, double>;

  /*!
   * \brief `Σ x[i] · y[i]` over native vectors: four of them accumulate independently, lane by lane, then
   *        their lanes and the remainder are summed in a fixed order. A single running sum waits on each
   *        addition before the next; this sum is as deterministic, in another order.
   */
  template <native_scalar T>
  [[nodiscard]] T dot_product(const T*    x,
                              const T*    y,
                              std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t   width {native_vector_bytes / sizeof(T)};
    constexpr std::size_t   lanes {4};
    std::array<lane, lanes> sum   {};
    const std::size_t       done  {count - (count % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
#  pragma GCC unroll 4 // the sums in registers: GCC keeps a rolled array of vectors in memory
      for (std::size_t l {0}; l < lanes; ++l) {
        lane xv;
        lane yv;
        std::memcpy(&xv, x + i + (l * width), sizeof(lane));
        std::memcpy(&yv, y + i + (l * width), sizeof(lane));
        sum[l] += xv * yv;
      }
    }
    T total {};
    for (std::size_t l {0}; l < lanes; ++l) {
      for (std::size_t k {0}; k < width; ++k) {
        total += sum[l][k];
      }
    }
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    return total;
  }

  /*!
   * \brief The sums of `Σ x[i] · y[i]` of complex numbers over native vectors of their parts, `{re·re, im·im, re·im,
   *        im·re}`: four vectors accumulate `x · y` part by part
   *        (`re·re` in the even lanes, `im·im` in the odd), four `x` times `y` with its parts swapped (`re·im`,
   *        `im·re`); their lanes, then the remainder's products, summed in a fixed order, the real part their first
   *        two sums' difference, the imaginary part the other two's sum. Where a running complex sum waited on each
   *        addition, and each product tested its parts for NaN (C Annex G): six AVX2 cores, a 189 × 206 complex
   *        matrix times a vector, 58 µs (NumPy 3.5).
   */
  template <split_complex C>
  [[nodiscard]] std::array<typename C::value_type, 4> complex_dot_sums(const C*    x,
                                                                       const C*    y,
                                                                       std::size_t count)
  {
    using R    = typename C::value_type;
    using lane = typename native_vector<R>::type;
    constexpr std::size_t   width {native_vector_bytes / sizeof(R)}; // parts in a vector: an even count
    constexpr std::size_t   lanes {4};
    const auto* const       a     {reinterpret_cast<const R*>(x)};   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is an array of its two parts
    const auto* const       b     {reinterpret_cast<const R*>(y)};   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): as above
    const std::size_t       parts {2 * count};
    std::array<lane, lanes> same  {};
    std::array<lane, lanes> cross {};
    const std::size_t       done  {parts - (parts % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
#  pragma GCC unroll 4 // the sums in registers
      for (std::size_t l {0}; l < lanes; ++l) {
        lane av;
        lane bv;
        std::memcpy(&av, a + i + (l * width), sizeof(lane));
        std::memcpy(&bv, b + i + (l * width), sizeof(lane));
        std::array<R, width> pairs_swapped {}; // each pair's parts swapped, then loaded: GCC 15 takes a lane set element by element for uninitialised
        for (std::size_t k {0}; k < width; ++k) {
          pairs_swapped[k] = bv[k ^ 1U];
        }
        lane swapped;
        std::memcpy(&swapped, pairs_swapped.data(), sizeof(lane));
        same[l]  += av * bv;
        cross[l] += av * swapped;
      }
    }
    R rr {};
    R ii {};
    R ri {};
    R ir {};
    for (std::size_t l {0}; l < lanes; ++l) {
      for (std::size_t k {0}; k < width; k += 2) {
        rr += same[l][k];
        ii += same[l][k + 1];
        ri += cross[l][k];
        ir += cross[l][k + 1];
      }
    }
    for (std::size_t i {done}; i < parts; i += 2) {
      rr += a[i] * b[i];
      ii += a[i + 1] * b[i + 1];
      ri += a[i] * b[i + 1];
      ir += a[i + 1] * b[i];
    }
    return {rr, ii, ri, ir};
  }

  //! \brief `Σ x[i] · y[i]` of complex numbers: \ref complex_dot_sums' `re·re − im·im` and `re·im + im·re`.
  template <split_complex C>
  [[nodiscard]] C dot_product(const C*    x,
                              const C*    y,
                              std::size_t count)
  {
    const auto [rr, ii, ri, ir] {complex_dot_sums(x, y, count)};
    return C {rr - ii, ri + ir};
  }

  //! \brief `Σ conj(x[i]) · y[i]` of complex numbers: the same sums, \ref complex_dot_sums', as `re·re + im·im` and
  //!        `re·im − im·re`, where a running complex sum waited on each addition and each product's rescue.
  template <split_complex C>
  [[nodiscard]] C conjugate_dot_product(const C*    x,
                                        const C*    y,
                                        std::size_t count)
  {
    const auto [rr, ii, ri, ir] {complex_dot_sums(x, y, count)};
    return C {rr + ii, ri - ir};
  }

  /*!
   * \brief \ref dot_product of each of \p rows rows of \p m (\p ld apart) with \p x, into \p out: four rows at a time,
   *        each summed exactly as \ref dot_product sums it (its lanes, their reduction, its remainder), so each is
   *        that dot product to the bit; the four share each load of \p x, and their sixteen chains keep the units
   *        busy where one row's four waited on their last addition (a matrix of order 256 times a vector on one
   *        arm64 core: 10.8 → 8.2 µs).
   */
  template <native_scalar T>
  void row_dots(const T*    m,
                std::size_t ld,
                std::size_t rows,
                const T*    x,
                std::size_t count,
                T*          out)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    constexpr std::size_t lanes {4};
    constexpr std::size_t group {4};
    const std::size_t     done  {count - (count % (width * lanes))};
    std::size_t           r0    {0};
    for (; r0 + group <= rows; r0 += group) {
      std::array<std::array<lane, lanes>, group> sum {};
      for (std::size_t i {0}; i < done; i += width * lanes) {
#  pragma GCC unroll 4 // the sums in registers: GCC keeps a rolled array of vectors in memory
        for (std::size_t l {0}; l < lanes; ++l) {
          lane xv;
          std::memcpy(&xv, x + i + (l * width), sizeof(lane));
#  pragma GCC unroll 4
          for (std::size_t r {0}; r < group; ++r) {
            lane mv;
            std::memcpy(&mv, m + ((r0 + r) * ld) + i + (l * width), sizeof(lane));
            sum[r][l] += mv * xv;
          }
        }
      }
      for (std::size_t r {0}; r < group; ++r) {
        T total {};
        for (std::size_t l {0}; l < lanes; ++l) {
          for (std::size_t k {0}; k < width; ++k) {
            total += sum[r][l][k];
          }
        }
        const T* const row {m + ((r0 + r) * ld)};
        for (std::size_t i {done}; i < count; ++i) {
          total += row[i] * x[i];
        }
        out[r0 + r] = total;
      }
    }
    for (; r0 < rows; ++r0) {
      out[r0] = dot_product(m + (r0 * ld), x, count);
    }
  }

  /*!
   * \brief `Σ row[i] · v[i]`, and `out[i] += row[i] · f`, for \p count elements, in one pass over \p row: a
   *        symmetric matrix-vector product takes both from each row of its lower triangle, which two passes
   *        would read twice. The sum is accumulated as \ref dot_product accumulates it.
   */
  template <native_scalar T>
  [[nodiscard]] T symmetric_row(const T*    row,
                                const T*    v,
                                const T&    f,
                                T*          out,
                                std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t   width                  {native_vector_bytes / sizeof(T)};
    constexpr std::size_t   lanes                  {4};
    const T                 factor                 {f}; // read once: it may be an element of v
    const lane              splat  = factor - lane {};  // factor in every lane
    std::array<lane, lanes> sum                    {};
    const std::size_t       done                   {count - (count % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
#  pragma GCC unroll 4 // the sums in registers: GCC keeps a rolled array of vectors in memory
      for (std::size_t l {0}; l < lanes; ++l) {
        lane rv;
        lane vv;
        lane ov;
        std::memcpy(&rv, row + i + (l * width), sizeof(lane));
        std::memcpy(&vv, v + i + (l * width), sizeof(lane));
        std::memcpy(&ov, out + i + (l * width), sizeof(lane));
        sum[l] += rv * vv;
        ov     += rv * splat;
        std::memcpy(out + i + (l * width), &ov, sizeof(lane));
      }
    }
    T total {};
    for (std::size_t l {0}; l < lanes; ++l) {
      for (std::size_t k {0}; k < width; ++k) {
        total += sum[l][k];
      }
    }
    for (std::size_t i {done}; i < count; ++i) {
      total  += row[i] * v[i];
      out[i] += row[i] * factor;
    }
    return total;
  }

  /*!
   * \brief \ref symmetric_row for the \ref symmetric_row_group rows `i…i+3` of a lower triangle at once, row `r` at
   *        `rows + r · n`: `out[j] += row_r[j] · v[i+r]` for `j < i + r`, the rows in order, and the sums `Σⱼ≤ᵢ₊ᵣ
   *        row_r[j] v[j]` returned. Their shared columns `j < i` go by native vectors, `v` and `out` read once for
   *        the group where each row would read them again; the rest by \ref symmetric_rows_tail.
   */
  template <native_scalar T>
  [[nodiscard]] std::array<T, symmetric_row_group> symmetric_rows(const T*    rows,
                                                                  std::size_t n,
                                                                  const T*    v,
                                                                  T*          out,
                                                                  std::size_t i)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t                 width {native_vector_bytes / sizeof(T)};
    std::array<lane, symmetric_row_group> sum   {};
    std::array<lane, symmetric_row_group> splat {};
    for (std::size_t r {0}; r < symmetric_row_group; ++r) {
      splat[r] = v[i + r] - lane {}; // v[i + r] in every lane
    }
    const std::size_t done {i - (i % width)};
    for (std::size_t j {0}; j < done; j += width) {
      lane vv;
      lane ov;
      std::memcpy(&vv, v + j, sizeof(lane));
      std::memcpy(&ov, out + j, sizeof(lane));
#  pragma GCC unroll 4 // the sums in registers: GCC keeps a rolled array of vectors in memory
      for (std::size_t r {0}; r < symmetric_row_group; ++r) {
        lane rv;
        std::memcpy(&rv, rows + (r * n) + j, sizeof(lane));
        sum[r] += rv * vv;
        ov     += rv * splat[r];
      }
      std::memcpy(out + j, &ov, sizeof(lane));
    }
    std::array<T, symmetric_row_group> sums {};
    for (std::size_t r {0}; r < symmetric_row_group; ++r) {
      for (std::size_t k {0}; k < width; ++k) {
        sums[r] += sum[r][k];
      }
    }
    return symmetric_rows_tail(rows, n, v, out, i, done, sums);
  }

  /*!
   * \brief The first largest magnitude of `x[0…count)` over native vectors, as the scalar search finds it: \p best
   *        and \p at come in as `|x[0]|` and 0, and leave as the largest of the elements this did and the first
   *        index holding it; returns how many leading elements it did (only `x[0]` under four vectors). Each lane of
   *        four independent chains keeps its own first strictly larger element, so the lane holding the largest at
   *        the smallest index is the scalar search's; a NaN is never larger, as there. The scalar search, a compare
   *        and a branch per element, was a fifth of an LU of order 256, and one chain waits a compare and a select
   *        per vector (one core, order 256: arm64 clang 341 → 306 µs, GCC 14 389 → 366; x86-64 GCC 13 −3 %).
   * \param[in]     x     The elements, at least one.
   * \param[in]     count Their number.
   * \param[in,out] best  The largest magnitude found.
   * \param[in,out] at    Its first index.
   * \return The leading elements searched.
   */
  template <native_scalar T>
  std::size_t largest_magnitude_vectors(const T*     x,
                                        std::size_t  count,
                                        T&           best,
                                        std::size_t& at)
  {
    using lane  = typename native_vector<T>::type;
    using mask  = decltype(lane {} > lane {});
    using index = std::remove_cvref_t<decltype(mask {}[0])>;
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    constexpr std::size_t ways  {4}; // independent chains: one waits a compare and a select per vector
    if (count < 1 + (ways * width)) {
      return 1;
    }
    const mask                magnitude {~std::bit_cast<mask>(T {-0.0} - lane {})}; // every bit but the sign's
    std::array<lane, ways>    top;
    std::array<mask, ways>    where     {};                                         // index 0 in every lane
    std::array<mask, ways>    next;                                                 // each chain's next indices
    for (std::size_t w {0}; w < ways; ++w) {
      top[w] = best - lane {}; // best in every lane
      for (std::size_t k {0}; k < width; ++k) {
        next[w][k] = static_cast<index>(1 + (w * width) + k);
      }
    }
    std::size_t i {1};
    for (; i + (ways * width) <= count; i += ways * width) { // selects by masks: each lane takes its own side
#  pragma GCC unroll 4                                       // the chains in registers: GCC keeps a rolled array of vectors in memory
      for (std::size_t w {0}; w < ways; ++w) {
        lane v;
        std::memcpy(&v, x + i + (w * width), sizeof(lane));
        const lane v_abs  {std::bit_cast<lane>(std::bit_cast<mask>(v) & magnitude)};
        const mask larger {v_abs > top[w]};
        top[w]    = std::bit_cast<lane>((std::bit_cast<mask>(v_abs) & larger) | (std::bit_cast<mask>(top[w]) & ~larger));
        where[w]  = (next[w] & larger) | (where[w] & ~larger);
        next[w]  += static_cast<index>(ways * width);
      }
    }
    for (std::size_t w {0}; w < ways; ++w) { // each lane holds its subsequence's first largest
      for (std::size_t k {0}; k < width; ++k) {
        const auto lane_at {static_cast<std::size_t>(where[w][k])};
        if (top[w][k] > best || (top[w][k] == best && lane_at < at)) {
          best = top[w][k];
          at   = lane_at;
        }
      }
    }
    return i;
  }

#endif

  //! \brief The rows of `y` \ref subtract_rows takes at once.
  inline constexpr std::size_t product_rows {8};

#if defined(__GNUC__) && !defined(__clang__)

  //! \brief The leading elements of `y[i] -= f · x[i]`, a whole number of native vectors: how many it did.
  template <native_scalar T>
  std::size_t subtract_scaled_vectors(T*          y,
                                      const T*    x,
                                      const T&    f,
                                      std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    const std::size_t     done  {count - (count % width)};
    for (std::size_t i {0}; i < done; i += width) {
      lane yv;
      lane xv;
      std::memcpy(&yv, y + i, sizeof(lane));
      std::memcpy(&xv, x + i, sizeof(lane));
      yv -= f * xv;
      std::memcpy(y + i, &yv, sizeof(lane));
    }
    return done;
  }

  /*!
   * \brief The leading elements of `y[i] -= Σₜ f[t] · x[(t · ldx) + i]`, a whole number of blocks of four native
   *        vectors held through the terms: how many it did. Each element takes its terms in ascending order, a
   *        product then a difference: GCC fuses no multiply-add of its vectors, though under `-ffp-contract=on` it
   *        fuses the scalar statement of the elements past them (arm64), so a row's elements may be rounded two ways,
   *        the split set by the count alone (the same bits on any number of threads). GCC leaves the scalar block of
   *        \ref subtract_products unvectorised; each lane moves in and out alone, as \ref gemm_kernel's do.
   */
  template <native_scalar T>
  std::size_t subtract_combined_vectors(T*          y,
                                        const T*    f,
                                        const T*    x,
                                        std::size_t ldx,
                                        std::size_t terms,
                                        std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t lanes {4};
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    const std::size_t     done  {count - (count % (lanes * width))};
    for (std::size_t i {0}; i < done; i += lanes * width) {
      std::array<lane, lanes> held {};
#  pragma GCC unroll 4
      for (std::size_t u {0}; u < lanes; ++u) {
        std::memcpy(&held[u], y + i + (u * width), sizeof(lane));
      }
      for (std::size_t t {0}; t < terms; ++t) {
        const T factor {f[t]};
#  pragma GCC unroll 4
        for (std::size_t u {0}; u < lanes; ++u) {
          lane row;
          std::memcpy(&row, x + (t * ldx) + i + (u * width), sizeof(lane));
          held[u] -= factor * row;
        }
      }
#  pragma GCC unroll 4
      for (std::size_t u {0}; u < lanes; ++u) {
        std::memcpy(y + i + (u * width), &held[u], sizeof(lane));
      }
    }
    return done;
  }

  /*!
   * \brief The leading columns of \ref subtract_rows, a whole number of native vectors, one per row held through the
   *        terms: how many it did. Each element takes its terms in ascending order and is rounded as
   *        \ref subtract_combined_vectors's are. GCC keeps the scalar block's rows in memory.
   */
  template <native_scalar T>
  std::size_t subtract_rows_vectors(T*          y,
                                    std::size_t ldy,
                                    const T*    f,
                                    std::size_t ldf,
                                    const T*    x,
                                    std::size_t ldx,
                                    std::size_t terms,
                                    std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    const std::size_t     done  {count - (count % width)};
    for (std::size_t i {0}; i < done; i += width) {
      std::array<lane, product_rows> held {};
#  pragma GCC unroll 8
      for (std::size_t r {0}; r < product_rows; ++r) {
        std::memcpy(&held[r], y + (r * ldy) + i, sizeof(lane));
      }
      for (std::size_t t {0}; t < terms; ++t) {
        lane row;
        std::memcpy(&row, x + (t * ldx) + i, sizeof(lane));
#  pragma GCC unroll 8
        for (std::size_t r {0}; r < product_rows; ++r) {
          const T factor {f[(r * ldf) + t]};
          held[r] -= factor * row;
        }
      }
#  pragma GCC unroll 8
      for (std::size_t r {0}; r < product_rows; ++r) {
        std::memcpy(y + (r * ldy) + i, &held[r], sizeof(lane));
      }
    }
    return done;
  }

  //! \brief The leading elements of `y[i] += f · x[i]`, a whole number of native vectors: how many it did.
  template <native_scalar T>
  std::size_t add_scaled_vectors(T*          y,
                                 const T*    x,
                                 const T&    f,
                                 std::size_t count)
  {
    using lane = typename native_vector<T>::type;
    constexpr std::size_t width {native_vector_bytes / sizeof(T)};
    const std::size_t     done  {count - (count % width)};
    for (std::size_t i {0}; i < done; i += width) {
      lane yv;
      lane xv;
      std::memcpy(&yv, y + i, sizeof(lane));
      std::memcpy(&xv, x + i, sizeof(lane));
      yv += f * xv;
      std::memcpy(y + i, &yv, sizeof(lane));
    }
    return done;
  }

  /*!
   * \brief The leading elements of `y[i] += f · x[i]` of complex numbers, a whole number of native vectors of their
   *        parts: how many it did. Each product's four part products rounded apart, then `re·re − im·im` and
   *        `re·im + im·re`, then added to `y` (\ref complex_scaled_add's arithmetic): its statements apart, so no
   *        multiply-add is fused, and the swapped parts' sign a multiplication by ±1, exact.
   */
  template <split_complex C>
  std::size_t add_scaled_vectors(C*          y,
                                 const C*    x,
                                 const C&    f,
                                 std::size_t count)
  {
    using R    = C::value_type;
    using lane = typename native_vector<R>::type;
    constexpr std::size_t width {native_vector_bytes / sizeof(R)}; // parts in a vector: an even count
    auto* const           out   {reinterpret_cast<R*>(y)};         // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is an array of its two parts
    const auto* const     in    {reinterpret_cast<const R*>(x)};   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): as above
    lane                  real_part;
    lane                  imaginary_part;
    lane                  sign;
    for (std::size_t k {0}; k < width; ++k) {
      real_part[k]      = f.real();
      imaginary_part[k] = f.imag();
      sign[k]           = (k % 2 == 0) ? R {-1} : R {1};
    }
    const std::size_t parts {2 * count};
    const std::size_t done  {parts - (parts % width)};
    for (std::size_t i {0}; i < done; i += width) {
      lane yv;
      lane xv;
      std::memcpy(&yv, out + i, sizeof(lane));
      std::memcpy(&xv, in + i, sizeof(lane));
      std::array<R, width> pairs_swapped {}; // as complex_dot_sums's
      for (std::size_t k {0}; k < width; ++k) {
        pairs_swapped[k] = xv[k ^ 1U];
      }
      lane swapped;
      std::memcpy(&swapped, pairs_swapped.data(), sizeof(lane));
      const lane straight       {real_part * xv};            // re·re, re·im
      const lane crossed        {imaginary_part * swapped};  // im·im, im·re
      const lane signed_crossed {sign * crossed};            // −im·im, +im·re
      const lane product        {straight + signed_crossed}; // re·re − im·im, re·im + im·re
      yv = yv + product;
      std::memcpy(out + i, &yv, sizeof(lane));
    }
    return done / 2;
  }

#endif

  /*!
   * \brief One tile of a matrix product kernel, \p rows × \p cols of C (at most \p Rows × \p Cols, leading dimension
   *        \p ldc), through \p full, which computes a whole tile in place given its first element and leading
   *        dimension: a whole tile in C itself, a partial one in a copy whose outside is zero. Only that copy is
   *        zeroed: zeroing it for every tile, whole ones included, took a twelfth of an x86-64 product's time.
   */
  template <std::size_t Rows,
            std::size_t Cols,
            typename T,
            typename Full>
  void through_edge(T*          c,
                    std::size_t ldc,
                    std::size_t rows,
                    std::size_t cols,
                    Full        full) noexcept
  {
    if (rows == Rows && cols == Cols) {
      full(c, ldc);
      return;
    }
    std::array<T, Rows * Cols> edge {};
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        edge[(i * Cols) + j] = c[(i * ldc) + j];
      }
    }
    full(edge.data(), Cols);
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        c[(i * ldc) + j] = edge[(i * Cols) + j];
      }
    }
  }

  //! \brief No leading elements in vectors, for a type no native vector holds.
  template <typename T>
  std::size_t subtract_scaled_vectors(T* /*y*/,
                                      const T* /*x*/,
                                      const T& /*f*/,
                                      std::size_t /*count*/)
  {
    return 0;
  }

  //! \brief Only `x[0]`, the one \p best and \p at come in with, for a type no native vector holds.
  template <typename T,
            typename Magnitude>
  std::size_t largest_magnitude_vectors(const T* /*x*/,
                                        std::size_t /*count*/,
                                        Magnitude& /*best*/,
                                        std::size_t& /*at*/)
  {
    return 1;
  }

  //! \brief No leading blocks in vectors, for a type no native vector holds or a compiler that vectorises the
  //!        scalar block itself (Clang).
  template <typename T>
  std::size_t subtract_combined_vectors(T* /*y*/,
                                        const T* /*f*/,
                                        const T* /*x*/,
                                        std::size_t /*ldx*/,
                                        std::size_t /*terms*/,
                                        std::size_t /*count*/)
  {
    return 0;
  }

  //! \brief No leading columns in vectors, for a type no native vector holds or a compiler that vectorises the
  //!        scalar block itself (Clang).
  template <typename T>
  std::size_t subtract_rows_vectors(T* /*y*/,
                                    std::size_t /*ldy*/,
                                    const T* /*f*/,
                                    std::size_t /*ldf*/,
                                    const T* /*x*/,
                                    std::size_t /*ldx*/,
                                    std::size_t /*terms*/,
                                    std::size_t /*count*/)
  {
    return 0;
  }

  //! \brief No leading elements in vectors, for a type no native vector holds.
  template <typename T>
  std::size_t add_scaled_vectors(T* /*y*/,
                                 const T* /*x*/,
                                 const T& /*f*/,
                                 std::size_t /*count*/)
  {
    return 0;
  }

  //! \brief `Σ row[i] · v[i]` and `out[i] += row[i] · f`, in order, for a type no native vector holds.
  template <typename T>
  [[nodiscard]] T symmetric_row(const T*    row,
                                const T*    v,
                                const T&    f,
                                T*          out,
                                std::size_t count)
  {
    const T factor {f}; // read once: it may be an element of v
    T       total  {};
    for (std::size_t i {0}; i < count; ++i) {
      total  += row[i] * v[i];
      out[i] += row[i] * factor;
    }
    return total;
  }

  //! \brief \ref symmetric_rows one column at a time, for a type no native vector holds.
  template <typename T>
  [[nodiscard]] std::array<T, symmetric_row_group> symmetric_rows(const T*    rows,
                                                                  std::size_t n,
                                                                  const T*    v,
                                                                  T*          out,
                                                                  std::size_t i)
  {
    return symmetric_rows_tail(rows, n, v, out, i, 0, {});
  }

  //! \brief `Σ x[i] · y[i]`, in order, for a type no native vector holds.
  template <typename T>
  [[nodiscard]] T dot_product(const T*    x,
                              const T*    y,
                              std::size_t count)
  {
    T total {};
    for (std::size_t i {0}; i < count; ++i) {
      total += x[i] * y[i];
    }
    return total;
  }

  //! \brief `Σ conj(x[i]) · y[i]`, in order, where no native vector holds the parts.
  template <typename T>
  [[nodiscard]] T conjugate_dot_product(const T*    x,
                                        const T*    y,
                                        std::size_t count)
  {
    T total {};
    for (std::size_t i {0}; i < count; ++i) {
      total += std::conj(x[i]) * y[i];
    }
    return total;
  }

  //! \brief \ref dot_product of each row of \p m with \p x, for a type no native vector holds.
  template <typename T>
  void row_dots(const T*    m,
                std::size_t ld,
                std::size_t rows,
                const T*    x,
                std::size_t count,
                T*          out)
  {
    for (std::size_t r {0}; r < rows; ++r) {
      out[r] = dot_product(m + (r * ld), x, count);
    }
  }

  /*!
   * \brief `a · b`, as `std::operator*` computes it, in line. libc++'s complex product (the four products and their
   *        sums as statements of their own, so unfused) is copied here, its rescue of a product whose parts are both
   *        NaN left to it: where it was a call, out of line in a translation unit as large as the Python extension,
   *        it took 45 % of a complex LU of order 48. Elsewhere (libstdc++, whose product the compiler expands) and for
   *        a real `T`, the operator itself.
   */
  template <typename T>
  [[nodiscard]] NUMERICAL_ALWAYS_INLINE T operator_times(const T& a,
                                                         const T& b)
  {
    return a * b;
  }

#if defined(_LIBCPP_VERSION)
  //! \brief \ref operator_times of two complex numbers under libc++: its product's statements, its rescue its own.
  template <typename R>
  [[nodiscard]] NUMERICAL_ALWAYS_INLINE std::complex<R> operator_times(const std::complex<R>& a,
                                                                       const std::complex<R>& b)
  {
    const R ac {a.real() * b.real()};
    const R bd {a.imag() * b.imag()};
    const R ad {a.real() * b.imag()};
    const R bc {a.imag() * b.real()};
    const R x  {ac - bd};
    const R y  {ad + bc};
    if (std::isnan(x) && std::isnan(y)) [[unlikely]] {
      return a * b;
    }
    return {x, y};
  }

#endif

  /*!
   * \brief `y[i] -= f · x[i]` for \p count elements of \p y and \p x, which do not overlap. Each element is
   *        computed exactly as that scalar statement computes it (its product rounded before its difference, or
   *        fused with it where the compiler contracts the statement); the arguments follow the expression, since
   *        a compiler may contract `f · x` and `x · f` of complex numbers differently. Under GCC, over native
   *        vectors where `T` has them: at `-O2` GCC leaves the scalar loop scalar, unable to rule out an overlap
   *        without a check it will not add. Clang vectorises the scalar loop itself, with that check, and better
   *        than these vectors (an LU solve of order 128 on arm64 took 19 % longer through them). For `double`,
   *        gemm.hpp's overload chooses the x86-64 units at run time, a fused multiply-add per element there.
   */
  template <typename T>
  void subtract_product(T*          y,
                        const T&    f,
                        const T*    x,
                        std::size_t count)
  {
    const T factor {f}; // read once, as a scalar loop would: it may be an element of y's array
    for (std::size_t i {subtract_scaled_vectors(y, x, factor, count)}; i < count; ++i) {
      y[i] -= operator_times(factor, x[i]);
    }
  }

  /*!
   * \brief `y[i] -= Σₜ f[t] · x[(t · ldx) + i]` for \p count elements of \p y, \p terms rows of \p x apart by \p ldx:
   *        \ref subtract_product for each term in ascending order, so each element is computed as that sequence of
   *        calls computes it. For `double`, gemm.hpp's overload holds the elements in registers through the terms.
   */
  template <typename T>
  void subtract_products(T*          y,
                         const T*    f,
                         const T*    x,
                         std::size_t ldx,
                         std::size_t terms,
                         std::size_t count)
  {
    for (std::size_t t {0}; t < terms; ++t) {
      subtract_product(y, f[t], x + (t * ldx), count);
    }
  }

  /*!
   * \brief \ref subtract_products for a real `T`: eight elements of \p y held in a local array through the terms,
   *        which the compiler keeps in registers, where each term was a call through the whole row. Each element
   *        takes its terms in the same order as a sequence of \ref subtract_product gives them (an LU solve of order
   *        64 and four columns on arm64: 18 → 8.5 µs), and the same value under Clang, and under GCC on x86-64, which
   *        fuses neither; under GCC on arm64 the vectors of \ref subtract_combined_vectors and
   *        \ref subtract_scaled_vectors round apart and the scalar statements fused, and their splits differ.
   */
  template <typename T>
  requires std::is_floating_point_v<T>
  void subtract_products(T*          y,
                         const T*    f,
                         const T*    x,
                         std::size_t ldx,
                         std::size_t terms,
                         std::size_t count)
  {
    std::size_t i {subtract_combined_vectors(y, f, x, ldx, terms, count)};
    // Blocks of 32, then of 8, 4, 2 and 1, each element a chain of its own through the terms: four multiply-adds a
    // cycle wait on their last one, and 8 elements held four chains (arm64, an inverse of order 128: 151 → 137 µs);
    // a narrower rest still takes the terms across its elements at once, where element by element each waited on
    // the last term (four columns of an LU solve of order 128 on arm64: 112 → 92 µs, the factorisation 75).
    const auto blocks = [&]<std::size_t Width>(std::integral_constant<std::size_t, Width>) {
                          for (; i + Width <= count; i += Width) {
                            std::array<T, Width> held {};
                            std::copy_n(y + i, Width, held.begin());
                            for (std::size_t t {0}; t < terms; ++t) {
                              const T        factor {f[t]};
                              const T* const row    {x + (t * ldx) + i};
                              for (std::size_t u {0}; u < Width; ++u) {
                                held[u] -= factor * row[u];
                              }
                            }
                            std::copy_n(held.begin(), Width, y + i);
                          }
                        };
    blocks(std::integral_constant<std::size_t, 32> {});
    blocks(std::integral_constant<std::size_t, 8> {});
    blocks(std::integral_constant<std::size_t, 4> {});
    blocks(std::integral_constant<std::size_t, 2> {});
    blocks(std::integral_constant<std::size_t, 1> {});
  }

  /*!
   * \brief \ref subtract_products for \ref product_rows rows of `y` at once, \p ldy apart, each with its own factors
   *        (\p ldf apart in \p f) on the same rows of \p x: `y_r[i] -= Σₜ f_r[t] · x[(t · ldx) + i]`. Each element takes
   *        the statement of \ref subtract_products, a term at a time in ascending order, so the same value; each element
   *        of `x` read serves every row, and the rows' chains of multiply-adds advance together, where a row alone
   *        waits on its last one through each term.
   */
  template <typename T>
  requires std::is_floating_point_v<T>
  void subtract_rows(T*          y,
                     std::size_t ldy,
                     const T*    f,
                     std::size_t ldf,
                     const T*    x,
                     std::size_t ldx,
                     std::size_t terms,
                     std::size_t count)
  {
    constexpr std::size_t width {4}; // four columns of eight rows: 32 values, which arm64's registers hold
    std::size_t           i     {subtract_rows_vectors(y, ldy, f, ldf, x, ldx, terms, count)};
    const auto            columns = [&]<std::size_t Width>(std::integral_constant<std::size_t, Width>) {
                                      for (; i + Width <= count; i += Width) {
                                        std::array<std::array<T, Width>, product_rows> held {};
                                        for (std::size_t r {0}; r < product_rows; ++r) {
                                          std::copy_n(y + (r * ldy) + i, Width, held[r].begin());
                                        }
                                        for (std::size_t t {0}; t < terms; ++t) {
                                          const T* const row {x + (t * ldx) + i};
                                          for (std::size_t r {0}; r < product_rows; ++r) {
                                            const T factor {f[(r * ldf) + t]};
                                            for (std::size_t u {0}; u < Width; ++u) {
                                              held[r][u] -= factor * row[u];
                                            }
                                          }
                                        }
                                        for (std::size_t r {0}; r < product_rows; ++r) {
                                          std::copy_n(held[r].begin(), Width, y + (r * ldy) + i);
                                        }
                                      }
                                    };
    columns(std::integral_constant<std::size_t, width> {});
    columns(std::integral_constant<std::size_t, 1> {});
  }

  //! \brief `y[i] -= x[i] · f`, as the other \ref subtract_product (a real product is the same either way round).
  template <typename T>
  void subtract_product(T*          y,
                        const T*    x,
                        const T&    f,
                        std::size_t count)
  {
    const T factor {f}; // read once, as a scalar loop would: it may be an element of y's array
    for (std::size_t i {subtract_scaled_vectors(y, x, factor, count)}; i < count; ++i) {
      y[i] -= operator_times(x[i], factor);
    }
  }

  //! \brief `y += f · x` of one complex number, its part products rounded apart as libc++'s operator computes them
  //!        (\ref add_scaled_vectors's arithmetic, for the elements past its vectors).
  template <split_complex C>
  void complex_scaled_add(C&       y,
                          const C& f,
                          const C& x)
  {
    using R = C::value_type;
    const R ac {f.real() * x.real()};
    const R bd {f.imag() * x.imag()};
    const R ad {f.real() * x.imag()};
    const R bc {f.imag() * x.real()};
    const R re {ac - bd};
    const R im {ad + bc};
    y = C {y.real() + re, y.imag() + im};
  }

  //! \brief \ref add_product of complex numbers: vectors of their parts, then \ref complex_scaled_add, every element
  //!        the same arithmetic.
  template <split_complex C>
  void add_product(C*          y,
                   const C&    f,
                   const C*    x,
                   std::size_t count)
  {
    const C factor {f}; // read once, as a scalar loop would: it may be an element of y's array
    for (std::size_t i {add_scaled_vectors(y, x, factor, count)}; i < count; ++i) {
      complex_scaled_add(y[i], factor, x[i]);
    }
  }

  //! \brief `y += f · x` of one element, as \ref add_product computes each of its elements.
  template <typename T>
  void add_one_product(T&       y,
                       const T& f,
                       const T& x)
  {
    y += operator_times(f, x);
  }

  //! \brief \ref add_one_product of complex numbers: \ref complex_scaled_add, as \ref add_product computes them.
  template <split_complex C>
  void add_one_product(C&       y,
                       const C& f,
                       const C& x)
  {
    complex_scaled_add(y, f, x);
  }

  //! \brief `y[i] += f · x[i]`, as \ref subtract_product.
  template <typename T>
  void add_product(T*          y,
                   const T&    f,
                   const T*    x,
                   std::size_t count)
  {
    const T factor {f}; // read once, as a scalar loop would: it may be an element of y's array
    for (std::size_t i {add_scaled_vectors(y, x, factor, count)}; i < count; ++i) {
      y[i] += operator_times(factor, x[i]);
    }
  }
} // namespace numerical::detail

#endif // NUMERICAL_SIMD_HPP
