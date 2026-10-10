// Downstream consumer smoke test: built by a separate CMake project that locates
// NumericAL only through `find_package(numerical CONFIG)` against an installed copy.
// It proves the installed config package, include paths, the Threads transitive
// dependency, and the numerical::numerical target all work end to end.
#include "numerical/numerical.hpp"

int main()
{
  const numerical::vector<double> v {1.0, 2.0, 3.0};
  return v.size() == 3 ? 0 : 1;
}
