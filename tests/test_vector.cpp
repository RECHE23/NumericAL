// The dense vector: construction/access, equality, element-wise arithmetic,
// scalar multiplication, dot product, tolerance comparison, and shape checks.
// Exercised over int (exact), double (tolerance), and complex (genericity).
//
// EXPECT(...) conditions containing braced `{a, b}` literals are wrapped in an
// extra pair of parentheses, so their commas are not seen as macro arguments.
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <thread>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::approx_equal;
  using numerical::dot;
  using numerical::vector;
} // namespace

TEST(construction_and_access)
{
  const vector<int> v {10, 20, 30};
  EXPECT_EQ(v.size(), 3U);
  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[2], 30);
  const vector<int> empty;
  EXPECT_EQ(empty.size(), 0U);
  const vector<int> from_buffer {std::vector<int> {1, 2}};
  EXPECT_EQ(from_buffer.size(), 2U);
  EXPECT_EQ(from_buffer[1], 2);
}

TEST(equality_is_size_and_element_wise)
{
  EXPECT((vector<int> {1, 2, 3} == vector<int> {1, 2, 3}));
  EXPECT((!(vector<int> {1, 2, 3} == vector<int> {1, 2, 4}))); // an element differs
  EXPECT((!(vector<int> {1, 2} == vector<int> {1, 2, 3})));    // a size differs
}

TEST(elementwise_add_and_subtract)
{
  EXPECT((vector<int> {1, 2, 3} + vector<int> {10, 20, 30} == vector<int> {11, 22, 33}));
  EXPECT((vector<int> {10, 20, 30} - vector<int> {1, 2, 3} == vector<int> {9, 18, 27}));
}

TEST(scalar_multiplication_on_either_side)
{
  EXPECT((3 * vector<int> {1, 2, 3} == vector<int> {3, 6, 9}));
  EXPECT((vector<int> {1, 2, 3} *3 == vector<int> {3, 6, 9}));
}

TEST(dot_product)
{
  EXPECT_EQ(dot(vector<int> {1, 2, 3}, vector<int> {4, 5, 6}), 32); // 4 + 10 + 18
  EXPECT_EQ(dot(vector<int> {}, vector<int> {}), 0);                // empty ⇒ zero
}

TEST(generic_over_the_scalar_complex)
{
  using c = std::complex<double>;
  const vector<c> v {c {0.0, 1.0}, c {1.0, 0.0}}; // [i, 1]
  EXPECT((dot(v, v) == c {0.0, 0.0}));            // i² + 1 = -1 + 1 = 0
  EXPECT((approx_equal(v + v, vector<c> {c {0.0, 2.0}, c {2.0, 0.0}})));
}

TEST(approx_equal_compares_within_a_tolerance)
{
  const vector<double> a {1.0, 2.0};
  EXPECT((approx_equal(a, vector<double> {1.0 + 1e-12, 2.0 - 1e-12})));         // within default tol
  EXPECT((!approx_equal(a, vector<double> {1.0, 2.5})));                        // outside tol
  EXPECT((!approx_equal(a, vector<double> {1.0})));                             // size mismatch
  const double nan {std::numeric_limits<double>::quiet_NaN()};
  const double inf {std::numeric_limits<double>::infinity()};
  EXPECT((!approx_equal(a, vector<double> {nan, 2.0})));                        // a NaN is close to nothing
  EXPECT((!approx_equal(vector<double> {nan}, vector<double> {nan})));          // not even a NaN
  EXPECT((approx_equal(vector<double> {inf, 1.0}, vector<double> {inf, 1.0}))); // equal infinities
  EXPECT((!approx_equal(vector<double> {inf}, vector<double> {-inf})));
}

TEST(approx_equal_compares_complex_vectors)
{
  using c = std::complex<double>;
  const vector<c> a {c {1.0, 1.0}, c {2.0, 0.0}};
  EXPECT((approx_equal(a, a)));                                                 // equal
  EXPECT((approx_equal(a, vector<c> {c {1.0, 1.0 + 1e-12}, c {2.0, 0.0}})));    // close
  EXPECT((!approx_equal(a, vector<c> {c {1.0, 1.5}, c {2.0, 0.0}})));           // far
  EXPECT((!approx_equal(a, vector<c> {c {1.0, 1.0}})));                         // size mismatch
}

TEST(shape_mismatch_throws)
{
  EXPECT_THROWS((vector<int> {1, 2} + vector<int> {1, 2, 3}), numerical::dimension_error);
  EXPECT_THROWS((vector<int> {1, 2} - vector<int> {1, 2, 3}), numerical::dimension_error);
  EXPECT_THROWS((dot(vector<int> {1, 2}, vector<int> {1, 2, 3})), numerical::dimension_error);
}

// A buffer from 1 MiB (under huge pages' bytes on Linux, 32 MiB elsewhere) takes a block of 4 KiB pages kept when it
// is freed: the next buffer it fits takes it back, its elements written anew, at another offset past the block's
// start; four blocks kept at most, the oldest freed when a fifth comes; threads asking for them at once each get one.
TEST(a_freed_buffer_from_one_mib_leaves_its_block_to_the_next)
{
  using allocator = numerical::detail::default_init_allocator<double>;
  using blocks    = numerical::detail::kept_page_blocks;
  const std::size_t least {numerical::detail::kept_pages_least / sizeof(double)};
  EXPECT(allocator::on_kept_pages(least) && !allocator::on_kept_pages(least - 1));
  EXPECT(!allocator::on_kept_pages(numerical::detail::kept_pages_largest / sizeof(double)));
  EXPECT(!allocator::on_kept_pages(std::numeric_limits<std::size_t>::max() / 4));
  // Just under the largest, the block's bytes rounded to its pages pass it: freed, not kept. From the largest on,
  // std::allocator's (on Linux, huge pages).
  for (const std::size_t edge : {(numerical::detail::kept_pages_largest / sizeof(double)) - 1, numerical::detail::kept_pages_largest / sizeof(double)}) {
    const numerical::buffer<double> held(edge, 5.0);
    EXPECT(held[0] == 5.0 && held[edge - 1] == 5.0);
  }
  constexpr std::size_t count {((std::size_t {3} << 20U) / sizeof(double)) + 777}; // no other test frees a block this size
  const auto            start {[](const double* at) { return reinterpret_cast<std::uintptr_t>(at) / blocks::alignment * blocks::alignment; }};
  std::uintptr_t        before {0};
  std::uintptr_t        offset {0};
  {
    const numerical::buffer<double> a(count, 2.0);
    before = start(a.data());
    offset = reinterpret_cast<std::uintptr_t>(a.data()) - before;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % 64, 0U);
  }
  const numerical::buffer<double> b(count, 3.0);
  EXPECT_EQ(start(b.data()), before);
  EXPECT(reinterpret_cast<std::uintptr_t>(b.data()) - before != offset);
  EXPECT(b[0] == 3.0 && b[count - 1] == 3.0);
  // The class itself: four places, the one kept longest freed for a fifth block.
  const std::size_t       mib {std::size_t {1} << 20U};
  blocks                  kept;
  std::size_t             got {0};
  std::vector<std::byte*> fresh;
  for (std::size_t k {1}; k <= 5; ++k) {
    fresh.push_back(static_cast<std::byte*>(::operator new(k * mib, std::align_val_t {blocks::alignment})));
    EXPECT(kept.keep(fresh.back(), k * mib));
  }
  EXPECT(kept.take(mib, got) == nullptr); // the 1 MiB block, kept longest, was freed
  EXPECT(kept.take(2 * mib, got) == fresh[1] && got == 2 * mib);
  EXPECT(kept.take(5 * mib, got) == fresh[4]);
  EXPECT(!kept.keep(fresh[1], blocks::largest));
  for (std::byte* const at : {fresh[1], fresh[4]}) {
    ::operator delete(at, std::align_val_t {blocks::alignment});
  }
  std::vector<std::thread> threads;
  std::vector<int>         held(6, 0);
  for (std::size_t t {0}; t < held.size(); ++t) {
    threads.emplace_back([&held, t] {
                           bool whole {true};
                           for (int round {0}; round < 16; ++round) {
                             numerical::buffer<double> c(count, static_cast<double>(t));
                             whole = whole && c[0] == static_cast<double>(t) && c[count - 1] == static_cast<double>(t);
                           }
                           held[t] = whole ? 1 : 0;
                         });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(held, (std::vector<int> {1, 1, 1, 1, 1, 1}));
}

#if defined(__linux__)
// A buffer of 16 MiB or more asks for huge pages: its data is there in full, written and read back, copied and grown,
// each one at another offset past its 2 MiB boundary; below, and for a count whose bytes overflow, std::allocator's.
TEST(a_large_buffer_on_huge_pages_holds_its_elements)
{
  using allocator = numerical::detail::default_init_allocator<double>;
  const std::size_t count {allocator::huge_least / sizeof(double)};
  EXPECT(allocator::on_huge_pages(count));
  EXPECT(!allocator::on_huge_pages(count - 1));
  EXPECT(!allocator::on_huge_pages(std::numeric_limits<std::size_t>::max() / 4));
  numerical::buffer<double> a(count);
  for (std::size_t i {0}; i < count; ++i) {
    a[i] = static_cast<double>(i);
  }
  numerical::buffer<double> b {a};
  b.resize(count + 5, 1.0);
  bool same                   {true};
  for (std::size_t i {0}; i < count; ++i) {
    same = same && b[i] == static_cast<double>(i);
  }
  EXPECT(same);
  EXPECT_EQ(b[count + 4], 1.0);
  const auto offset = [](const double* at) { return reinterpret_cast<std::uintptr_t>(at) % allocator::huge_page; };
  EXPECT(offset(a.data()) != offset(b.data()));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % alignof(std::max_align_t), 0U);
}
// Freed huge-page blocks are kept, two at most, the newest: a block is taken by an allocation it fits with no more than
// half of it unused, a third block freed frees the one kept longest, one of 64 MiB or more is not kept.
TEST(freed_huge_page_blocks_are_kept_for_the_next_that_fits)
{
  using blocks = numerical::detail::kept_huge_blocks;
  const std::size_t mib {std::size_t {1} << 20U};
  const auto        fresh {[](std::size_t bytes) { return static_cast<std::byte*>(::operator new(bytes, std::align_val_t {blocks::alignment})); }};
  const auto        free_block {[](std::byte* at) { ::operator delete(at, std::align_val_t {blocks::alignment}); }};
  blocks            kept;
  std::size_t       got    {0};
  EXPECT(kept.take(4 * mib, got) == nullptr);
  std::byte* const  first  {fresh(4 * mib)};
  std::byte* const  second {fresh(8 * mib)};
  std::byte* const  third  {fresh(6 * mib)};
  EXPECT(kept.keep(first, 4 * mib));
  EXPECT(kept.keep(second, 8 * mib));
  EXPECT(kept.take(9 * mib, got) == nullptr); // larger than either
  EXPECT(kept.take(3 * mib, got) == first && got == 4 * mib);
  EXPECT(kept.take(3 * mib, got) == nullptr); // the 8 MiB block would leave more than half unused
  EXPECT(kept.keep(first, 4 * mib));
  EXPECT(kept.keep(third, 6 * mib));          // both taken: the 8 MiB block, kept longest, is freed
  EXPECT(kept.take(8 * mib, got) == nullptr);
  EXPECT(kept.take(5 * mib, got) == third && got == 6 * mib);
  EXPECT(kept.take(4 * mib, got) == first);
  EXPECT(!kept.keep(first, blocks::largest));
  free_block(first);
  free_block(third);
}

// A large buffer freed leaves its block among the kept ones, and the next buffer it fits takes it back, its elements
// written anew; threads freeing and asking for them at once each get a block of their own.
TEST(a_large_buffer_takes_a_freed_block_back)
{
  using allocator = numerical::detail::default_init_allocator<double>;
  using blocks    = numerical::detail::kept_huge_blocks;
  constexpr std::size_t count {((allocator::huge_least * 5 / 2) / sizeof(double)) + 12345}; // no other test frees a block this size
  const auto            start {[](const double* at) { return reinterpret_cast<std::uintptr_t>(at) / allocator::huge_page * allocator::huge_page; }};
  std::uintptr_t        before {0};
  {
    const numerical::buffer<double> a(count, 2.0);
    before = start(a.data());
  }
  std::size_t      got   {0};
  std::byte* const block {blocks::shared().take(count * sizeof(double), got)};
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block), before);
  EXPECT(blocks::shared().keep(block, got));
  const numerical::buffer<double> b(count, 3.0);
  EXPECT_EQ(start(b.data()), before);
  EXPECT_EQ(b[count - 1], 3.0);
  std::vector<std::thread> threads;
  std::vector<int>         held(4, 0);
  for (std::size_t t {0}; t < held.size(); ++t) {
    threads.emplace_back([&held, t] {
                           bool whole {true};
                           for (int round {0}; round < 8; ++round) {
                             numerical::buffer<double> c(count, static_cast<double>(t));
                             whole = whole && c[0] == static_cast<double>(t) && c[count - 1] == static_cast<double>(t);
                           }
                           held[t] = whole ? 1 : 0;
                         });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(held, (std::vector<int> {1, 1, 1, 1}));
}
#endif
