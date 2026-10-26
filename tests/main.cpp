#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

#include <sciforge/test/framework.hpp>

namespace {
  // Keeps shard `i` of `n` alone among the registered tests (`shard` = "i/n"): every n-th test from the i-th, so that
  // n processes share the suite, each test in one of them. False where `shard` is not two counts, i below n.
  bool keep_shard(std::string_view shard)
  {
    const std::size_t slash  {std::min(shard.find('/'), shard.size())};
    const char* const begin  {shard.data()};
    const char* const middle {begin + slash};
    const char* const end    {begin + shard.size()};
    std::size_t       index  {0};
    std::size_t       count  {0};
    const auto        first  {std::from_chars(begin, middle, index)};
    if (first.ec != std::errc {} || first.ptr != middle || middle == end) {
      return false;
    }
    const auto second {std::from_chars(middle + 1, end, count)};
    if (second.ec != std::errc {} || second.ptr != end || index >= count) {
      return false;
    }
    std::vector<test::test_case> kept;
    for (std::size_t t {index}; t < test::registry().size(); t += count) {
      kept.push_back(test::registry()[t]);
    }
    test::registry() = kept;
    return true;
  }
} // namespace

// The tests whose name holds the first argument, or all of them; of one shard alone where that argument is
// "--shard=i/n", the filter then the second.
int main(int    argc,
         char** argv)
{
  const std::span<char*>      arguments {argv, static_cast<std::size_t>(argc)};
  constexpr std::string_view  prefix    {"--shard="};
  if (arguments.size() > 1 && std::string_view {arguments[1]}.starts_with(prefix)) {
    if (!keep_shard(std::string_view {arguments[1]}.substr(prefix.size()))) {
      (void)std::fputs("--shard= takes i/n, i below n\n", stderr);
      return 2;
    }
    return test::run_all(arguments.size() > 2 ? std::string_view {arguments[2]} : std::string_view {});
  }
  return test::run_all(argc, argv);
}
