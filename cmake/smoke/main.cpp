// Downstream consumer smoke test: built by a separate CMake project that locates NumericAL only through
// `find_package(numerical CONFIG)` against an installed copy, so the installed config package, include paths, the
// Threads dependency and the numerical::numerical target work end to end. It is also README.md's C++ example.
#include <cmath>
#include <complex>
#include <cstdio>

#include <numerical/numerical.hpp>

int main()
{
  const numerical::matrix<double> a {{4.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}};
  const numerical::vector<double> b {1.0, 2.0, 3.0};

  const numerical::vector<double>              x       {numerical::solve(a, b)}; // LU, partial pivoting
  const numerical::symmetric_eigen             eigen   {a};                      // A = V L V^T, as numpy.linalg.eigh
  const numerical::eigen_decomposition<double> general {a};                      // as numpy.linalg.eig
  const numerical::svd_decomposition<double>   svd     {a};                      // A = U S V^H

  std::printf("x = %.6f %.6f %.6f\n", x[0], x[1], x[2]);
  std::printf("eigh: %.6f %.6f %.6f\n", eigen.values()[0], eigen.values()[1], eigen.values()[2]);
  std::printf("eig: %.6f%+.6fi, its vectors %zu x %zu\n", general.values()[0].real(), general.values()[0].imag(),
              general.vectors().rows(), general.vectors().cols());
  std::printf("svd: %.6f %.6f %.6f\n", svd.singular_values()[0], svd.singular_values()[1], svd.singular_values()[2]);
  const numerical::vector<double> residual {(a * x) - b};
  return std::abs(residual[0]) + std::abs(residual[1]) + std::abs(residual[2]) < 1e-12 ? 0 : 1;
}
