// A race planted in a parallel_for: every chunk adds to one counter, unguarded, from the pool's workers. `make tsan`
// requires ThreadSanitizer to report it before it takes a clean run of the suite for an absence of races: a runtime
// that cannot map its shadow memory, or a build without the instrumentation, reports nothing on anything. Each chunk
// waits a millisecond first, so that the workers take some: chunks quick enough for the caller to run them all, before
// a worker woke, left no race to report (it happened once in a gate).
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <thread>

#include "numerical/numerical.hpp"

int main()
{
  std::size_t counter {0};
  numerical::detail::parallel_for(64, std::size_t {1} << 20U, [&counter](std::size_t /*chunk*/) {
                                    std::this_thread::sleep_for(std::chrono::milliseconds {1});
                                    for (std::size_t i {0}; i < 1000; ++i) {
                                      ++counter;
                                    }
                                  });
  (void)std::printf("%zu\n", counter); // NOLINT(cppcoreguidelines-pro-type-vararg): the count, kept alive
  return 0;
}
