// A gallery of classical hard matrices through the C++ decompositions, at each scalar the library takes, which the
// Python binding never reaches (it computes in double): LU, Cholesky, QR and the SVD (each method) in float, double and
// long double, at scales from the type's subnormal numbers to near its largest; the eigendecompositions and the Schur
// form in double. Each answer finite (or an explicit error where the matrix has none), within a backward error of order
// n ε ‖A‖ of the input, its factors orthonormal. svd_decomposition<float> of a matrix of rank 1 did not converge, unseen
// by any test. Under glibc, the same decompositions of the same finite inputs run once more with FE_INVALID trapping:
// none may produce a NaN on the way, the trap stopping at the operation that does.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#  include <fenv.h>
#endif

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;

  //! \brief A family: its name and its matrix of order n, real entries.
  struct family
  {
    std::string                                       name;
    std::function<std::vector<double>(std::size_t n)> make;
    std::vector<std::size_t>                          orders;
    bool                                              definite {false}; //!< positive definite, for Cholesky
  };

  std::vector<double> filled(std::size_t                                            n,
                             const std::function<double(std::size_t, std::size_t)>& entry)
  {
    std::vector<double> a(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        a[(i * n) + j] = entry(i, j);
      }
    }
    return a;
  }

  std::vector<family> gallery()
  {
    const auto skew = [](std::size_t n) {
                        return filled(n, [](std::size_t i, std::size_t j) {
                                        if (j == i + 1) {
                                          return 1.0;
                                        }
                                        return i == j + 1 ? -1.0 : 0.0;
                                      });
                      };
    const auto wilkinson = [](std::size_t n) {
                             const double m {static_cast<double>(n - 1) / 2.0};
                             return filled(n, [m](std::size_t i, std::size_t j) {
                                             if (i == j) {
                                               return std::abs(static_cast<double>(i) - m);
                                             }
                                             return i + 1 == j || j + 1 == i ? 1.0 : 0.0;
                                           });
                           };
    const auto rank_one = [](std::size_t n) {
                            return filled(n, [](std::size_t i, std::size_t j) {
                                            return static_cast<double>(static_cast<int>(i % 5) - 2) * static_cast<double>((j % 3) + 1);
                                          });
                          };
    const auto kahan = [](std::size_t n) {
                         const double s {std::sin(1.2)};
                         const double c {std::cos(1.2)};
                         return filled(n, [s, c](std::size_t i, std::size_t j) {
                                         return j < i ? 0.0 : std::pow(s, static_cast<double>(i)) * (i == j ? 1.0 : -c);
                                       });
                       };
    const auto frank = [](std::size_t n) {
                         return filled(n, [n](std::size_t i, std::size_t j) {
                                         return j + 1 >= i ? static_cast<double>(n - std::max(i, j)) : 0.0;
                                       });
                       };
    const auto minij = [](std::size_t n) {
                         return filled(n, [](std::size_t i, std::size_t j) { return static_cast<double>(std::min(i, j) + 1); });
                       };
    const auto grcar = [](std::size_t n) {
                         return filled(n, [](std::size_t i, std::size_t j) {
                                         if (i == j + 1) {
                                           return -1.0;
                                         }
                                         return j >= i && j <= i + 3 ? 1.0 : 0.0;
                                       });
                       };
    const auto shifted = [wilkinson](std::size_t n) {
                           std::vector<double> a {wilkinson(n)};
                           for (std::size_t i {0}; i < n; ++i) {
                             a[(i * n) + i] += 6.0;
                           }
                           return a;
                         };
    return {
      {.name = "skew tridiagonal", .make = skew, .orders = {2, 23, 55}},
      {.name = "Wilkinson", .make = wilkinson, .orders = {3, 21}},
      {.name = "rank 1", .make = rank_one, .orders = {2, 7, 40}},
      {.name = "Kahan", .make = kahan, .orders = {5, 30}},
      {.name = "Frank", .make = frank, .orders = {5, 20}},
      {.name = "Grcar", .make = grcar, .orders = {10, 40}},
      {.name = "ones", .make = [](std::size_t n) { return std::vector<double>(n * n, 1.0); }, .orders = {3, 40}},
      {.name = "zero", .make = [](std::size_t n) { return std::vector<double>(n * n, 0.0); }, .orders = {1, 4}},
      {.name = "min(i, j) + 1", .make = minij, .orders = {7, 40}, .definite = true},
      {.name = "Wilkinson + 6 I", .make = shifted, .orders = {21}, .definite = true},
    };
  }

  //! \brief The scales of a type: 1, a power of two halfway to its smallest normal number, one among its subnormal
  //!        numbers, and one a few hundred binades below its largest.
  template <typename T>
  std::vector<T> scales()
  {
    using limits = std::numeric_limits<T>;
    return {T {1}, std::ldexp(T {1}, limits::min_exponent / 2), std::ldexp(T {1}, limits::min_exponent - (limits::digits / 2)),
            std::ldexp(T {1}, (limits::max_exponent * 3) / 4)};
  }

  template <typename T>
  matrix<T> scaled(const std::vector<double>& entries,
                   std::size_t                n,
                   T                          scale)
  {
    std::vector<T> a(n * n);
    for (std::size_t e {0}; e < n * n; ++e) {
      a[e] = static_cast<T>(entries[e]) * scale;
    }
    return matrix<T> {n, n, a};
  }

  template <typename T>
  long double largest(const matrix<T>& a)
  {
    long double m {0};
    for (const T& x : a) {
      m = std::max(m, static_cast<long double>(std::abs(x)));
    }
    return m;
  }

  template <typename T>
  bool finite(const matrix<T>& a)
  {
    return std::ranges::all_of(a, [](const T& x) { return std::isfinite(static_cast<long double>(x)); });
  }

  //! \brief `‖a b − c‖` element by element, in long double.
  template <typename T>
  long double residual(const matrix<T>& a,
                       const matrix<T>& b,
                       const matrix<T>& c)
  {
    long double m {0};
    for (std::size_t i {0}; i < a.rows(); ++i) {
      for (std::size_t j {0}; j < b.cols(); ++j) {
        long double sum {0};
        for (std::size_t k {0}; k < a.cols(); ++k) {
          sum += static_cast<long double>(a(i, k)) * static_cast<long double>(b(k, j));
        }
        m = std::max(m, std::abs(sum - static_cast<long double>(c(i, j))));
      }
    }
    return m;
  }

  //! \brief `‖qᵀ q − I‖`, in long double.
  template <typename T>
  long double off_orthonormal(const matrix<T>& q)
  {
    long double m {0};
    for (std::size_t i {0}; i < q.cols(); ++i) {
      for (std::size_t j {0}; j < q.cols(); ++j) {
        long double sum {0};
        for (std::size_t r {0}; r < q.rows(); ++r) {
          sum += static_cast<long double>(q(r, i)) * static_cast<long double>(q(r, j));
        }
        m = std::max(m, std::abs(sum - (i == j ? 1.0L : 0.0L)));
      }
    }
    return m;
  }

  template <typename T>
  matrix<T> transposed(const matrix<T>& a)
  {
    return numerical::transpose(a);
  }

  //! \brief The bound of a backward error: 64 n ε ‖A‖ in `epsilon`, and the subnormal spacing of `T` the entries hold.
  template <typename T>
  long double bound(std::size_t n,
                    long double epsilon,
                    long double size)
  {
    return (64.0L * static_cast<long double>(n) * epsilon * size)
           + (1e3L * static_cast<long double>(n) * static_cast<long double>(std::numeric_limits<T>::denorm_min()));
  }

  //! \brief LU, QR, Cholesky and each SVD of every member of the gallery in `T`: the failures' descriptions.
  template <typename T>
  std::vector<std::string> generic_failures()
  {
    std::vector<std::string> failures;
    const long double        own         {std::numeric_limits<T>::epsilon()};
    const long double        svd_epsilon {std::max(own, static_cast<long double>(std::numeric_limits<double>::epsilon()))}; // carried in double
    for (const family& f : gallery()) {
      for (const std::size_t n : f.orders) {
        for (const T scale : scales<T>()) {
          const matrix<T>                      a {scaled(f.make(n), n, scale)};
          const long double                    size {largest(a)};
          const std::string                    label {f.name + " " + std::to_string(n) + " at 2^" + std::to_string(std::ilogb(scale))};
          const auto                           fail = [&](const std::string& what) {
                                                        std::string text {label};
                                                        text += ": ";
                                                        text += what;
                                                        failures.push_back(std::move(text));
                                                      };
          const numerical::qr_decomposition<T> qr {a};
          if (!finite(qr.q()) || off_orthonormal(qr.q()) > 64.0L * static_cast<long double>(n) * own
              || residual(qr.q(), qr.r(), a) > bound<T>(n, own, size)) {
            fail("qr");
          }
          for (const auto method : {numerical::svd_method::divide_and_conquer, numerical::svd_method::bidiagonal_qr, numerical::svd_method::jacobi}) {
            try {
              const numerical::svd_decomposition<T> s  {a, method};
              matrix<T>                             us {s.u()};
              std::vector<T>                        e(us.begin(), us.end());
              for (std::size_t i {0}; i < n; ++i) {
                for (std::size_t j {0}; j < us.cols(); ++j) {
                  e[(i * us.cols()) + j] *= static_cast<T>(s.singular_values()[j]);
                }
              }
              us = matrix<T> {n, us.cols(), e};
              if (!finite(s.u()) || !finite(s.v()) || off_orthonormal(s.u()) > 64.0L * static_cast<long double>(n) * svd_epsilon
                  || off_orthonormal(s.v()) > 64.0L * static_cast<long double>(n) * svd_epsilon
                  || residual(us, transposed(s.v()), a) > bound<T>(n, svd_epsilon, size)) {
                fail("svd " + std::to_string(static_cast<int>(method)));
              }
            }
            catch (const numerical::argument_error&) {
              // a long double matrix whose singular values lie past double's range (the SVD computes in double),
              // refused; any other type's is a defect
              if (!std::is_same_v<T, long double> || std::numeric_limits<long double>::max_exponent <= std::numeric_limits<double>::max_exponent) {
                fail("svd refused");
              }
            }
            catch (const numerical::error& e) {
              fail(std::string {"svd raised "} + e.what());
            }
          }
          try {
            const numerical::lu_decomposition<T> lu {a};
            if (!lu.singular()) {
              const numerical::vector<T> b(std::vector<T>(n, size > 0.0L ? static_cast<T>(size) : T {1}));
              const numerical::vector<T> x                                                          {lu.solve(b)};
              long double                worst                                                      {0};
              long double                x_size                                                     {0};
              for (std::size_t i {0}; i < n; ++i) {
                x_size = std::max(x_size, std::abs(static_cast<long double>(x[i])));
              }
              const numerical::vector<T> ax {a * x};
              for (std::size_t i {0}; i < n; ++i) {
                worst = std::max(worst, std::abs(static_cast<long double>(ax[i]) - static_cast<long double>(b[i])));
              }
              if (std::isfinite(x_size) && worst > bound<T>(n, own, (size * x_size) + static_cast<long double>(b[0])) + (1e3L * static_cast<long double>(n) * static_cast<long double>(std::numeric_limits<T>::denorm_min()) * x_size)) {
                fail("lu solve");
              }
            }
          }
          catch (const numerical::singular_matrix_error&) { // no solution: refused, as it should be
          }
          if (f.definite) {
            try {
              const numerical::cholesky_decomposition<T> c {a};
              if (!finite(c.lower()) || residual(c.lower(), transposed(c.lower()), a) > bound<T>(n, own, size)) {
                fail("cholesky");
              }
            }
            catch (const numerical::not_positive_definite_error&) {
              if (std::ilogb(scale) > std::numeric_limits<T>::min_exponent) { // definite no more only among the subnormals
                fail("cholesky refused");
              }
            }
          }
        }
      }
    }
    return failures;
  }

  //! \brief eig, Schur and each symmetric method in double of every member: the failures' descriptions.
  std::vector<std::string> double_failures()
  {
    std::vector<std::string> failures;
    const long double        eps {std::numeric_limits<double>::epsilon()};
    for (const family& f : gallery()) {
      for (const std::size_t n : f.orders) {
        for (const double scale : scales<double>()) {
          const matrix<double>                         a     {scaled(f.make(n), n, scale)};
          const long double                            size  {largest(a)};
          const std::string                            label {f.name + " " + std::to_string(n) + " at 2^" + std::to_string(std::ilogb(scale))};
          const numerical::schur_decomposition<double> s     {a};
          if (!finite(s.t()) || !finite(s.z()) || off_orthonormal(s.z()) > 64.0L * static_cast<long double>(n) * eps
              || residual(s.z(), matrix<double> {s.t() * transposed(s.z())}, a) > bound<double>(n, eps, size)) {
            failures.push_back(label + ": schur");
          }
          const numerical::eigen_decomposition<double> e {a};
          for (const auto& value : e.values()) {
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
              failures.push_back(label + ": eig values");
              break;
            }
          }
          if (f.definite) {
            for (const auto method : {numerical::eigen_method::divide_and_conquer, numerical::eigen_method::tridiagonal_ql, numerical::eigen_method::jacobi}) {
              const numerical::symmetric_eigen h {a, method};
              std::vector<double>              vl(n * n);
              for (std::size_t i {0}; i < n; ++i) {
                for (std::size_t j {0}; j < n; ++j) {
                  vl[(i * n) + j] = h.vectors()(i, j) * h.values()[j];
                }
              }
              if (off_orthonormal(h.vectors()) > 64.0L * static_cast<long double>(n) * eps
                  || residual(a, h.vectors(), matrix<double> {n, n, vl}) > bound<double>(n, eps, size)) {
                failures.push_back(label + ": eigh " + std::to_string(static_cast<int>(method)));
              }
            }
          }
        }
      }
    }
    return failures;
  }

  bool report(const std::vector<std::string>& failures)
  {
    for (const std::string& failure : failures) {
      static_cast<void>(std::fprintf(stderr, "  gallery: %s\n", failure.c_str()));
    }
    return failures.empty();
  }
} // namespace

TEST(the_gallery_in_float)
{
  EXPECT(report(generic_failures<float>()));
}

TEST(the_gallery_in_double)
{
  EXPECT(report(generic_failures<double>()));
  EXPECT(report(double_failures()));
}

TEST(the_gallery_in_long_double)
{
  EXPECT(report(generic_failures<long double>()));
}

#if defined(__GLIBC__)
TEST(the_gallery_makes_no_nan_from_finite_entries)
{
  // every invalid operation traps (SIGFPE) at the instruction: a NaN made on the way, even one discarded later, stops
  // the run where it is made
  const int before {fegetexcept()};
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID);
  const bool clean {report(generic_failures<double>()) && report(double_failures())};
  fedisableexcept(FE_ALL_EXCEPT);
  feenableexcept(before);
  EXPECT(clean);
}
#endif
