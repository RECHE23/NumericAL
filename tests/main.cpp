#include <sciforge/test/framework.hpp>

// The tests whose name holds the first argument, or all of them.
int main(int    argc,
         char** argv)
{
  return test::run_all(argc, argv);
}
