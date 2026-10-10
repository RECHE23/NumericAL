# Build system for NumericAL. Single source of truth for compilation: warning
# policy and the sanitizer/coverage build variants. The root Makefile is a thin
# orchestrator over these targets.
#
# NumericAL is a header-only leaf library: it depends on nothing but the C++
# standard library, and is generic over its scalar type.
cmake_minimum_required(VERSION 3.16)

# Single source of truth for the version: pyproject.toml (which `make release`
# bumps and setup.py reads). Deriving it here means the CMake package version can
# never drift from the Python package (the release template seds pyproject +
# __init__ only, so a hardcoded literal here would silently go stale).
file(READ "${CMAKE_CURRENT_LIST_DIR}/pyproject.toml" _numerical_pyproject)
if(NOT _numerical_pyproject MATCHES "[\r\n]version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"")
  message(FATAL_ERROR "NumericAL: could not parse the project version from pyproject.toml")
endif()
project(numerical VERSION "${CMAKE_MATCH_1}" LANGUAGES CXX)

# Fetches a header-only dependency's SOURCE, running none of its CMake. FetchContent_Populate(<name>),
# the source-only spelling, is deprecated from CMake 3.30 (CMP0169); MakeAvailable with a SOURCE_SUBDIR
# that holds no CMakeLists.txt fetches the same tree and adds nothing. The old spelling stays only for
# the 3.16/3.17 floor, which has no SOURCE_SUBDIR. A macro, so <name>_SOURCE_DIR lands in the caller.
include(FetchContent)
macro(numerical_fetch_source name repo tag)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        FetchContent_Declare(${name} GIT_REPOSITORY "${repo}" GIT_TAG "${tag}" SOURCE_SUBDIR no-cmake-here)
        FetchContent_MakeAvailable(${name})
    else()
        FetchContent_Declare(${name} GIT_REPOSITORY "${repo}" GIT_TAG "${tag}")
        FetchContent_GetProperties(${name})
        if(NOT ${name}_POPULATED)
            FetchContent_Populate(${name})
        endif()
    endif()
endmacro()

# Use ccache as a compiler launcher when available (a no-op otherwise): the
# coverage / sanitizer / dual-compiler gates rebuild from scratch, so caching
# object files makes repeat runs near-instant. Override with -DUSE_CCACHE=OFF.
option(USE_CCACHE "Use ccache as the compiler launcher when found" ON)
if(USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

option(NUMERICAL_SANITIZE "Build the tests with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(NUMERICAL_COVERAGE "Build the tests with LLVM source-based coverage instrumentation" OFF)

# Header-only library: consumers link numerical::numerical to get the include path.
# It also uses std::thread (parallel.hpp), so it carries the threads dependency.
find_package(Threads REQUIRED)
add_library(numerical INTERFACE)
add_library(numerical::numerical ALIAS numerical)
target_include_directories(numerical INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(numerical INTERFACE cxx_std_20)
target_link_libraries(numerical INTERFACE Threads::Threads)

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# find_package(numerical CONFIG) support: export the interface target and generate
# a config package, so a downstream CMake project can consume the installed
# library idiomatically:
#   find_package(numerical CONFIG REQUIRED)
#   target_link_libraries(app PRIVATE numerical::numerical)
install(TARGETS numerical EXPORT numericalTargets)
install(EXPORT numericalTargets
    NAMESPACE numerical::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerical)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/numericalConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/numericalConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerical)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/numericalConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/numericalConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/numericalConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerical)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()

enable_testing()

# The test harness (framework.hpp) is owned by SciForge — the ecosystem's shared
# infrastructure repository — and is never carried as a local copy. Resolution
# mirrors how this project finds its other sibling dependencies, two modes:
#  * Default (local development): a sibling checkout, used live. The maintainer's
#    layout uses ../sciforge; override with -DSCIFORGE_INCLUDE_DIR=/path.
#  * CI / reproducible builds: -DNUMERICAL_FETCH_SCIFORGE=ON fetches SciForge with
#    FetchContent from a configurable repository + tag. Prepared but inert by
#    default; once SciForge publishes a CalVer tag, pin SCIFORGE_TAG to it.
option(NUMERICAL_FETCH_SCIFORGE "Fetch SciForge (test harness) with FetchContent" OFF)
set(SCIFORGE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sciforge/include"
    CACHE PATH "Path to SciForge's include directory (maintainer's default; override for your layout)")
set(SCIFORGE_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../sciforge"
    CACHE STRING "SciForge git repository (path or URL) when fetching")
set(SCIFORGE_TAG "v2026.10.1" CACHE STRING "SciForge git tag when fetching (pin to a CalVer tag once released)")

if(NUMERICAL_FETCH_SCIFORGE)
    numerical_fetch_source(sciforge "${SCIFORGE_REPO}" "${SCIFORGE_TAG}")
    set(SCIFORGE_INCLUDE_DIR "${sciforge_SOURCE_DIR}/include")
endif()

if(NOT EXISTS "${SCIFORGE_INCLUDE_DIR}/sciforge/test/framework.hpp")
    message(FATAL_ERROR
        "SciForge test harness not found at SCIFORGE_INCLUDE_DIR='${SCIFORGE_INCLUDE_DIR}'. "
        "Use -DNUMERICAL_FETCH_SCIFORGE=ON, or set -DSCIFORGE_INCLUDE_DIR=/path/to/sciforge/include.")
endif()

file(GLOB NUMERICAL_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
add_executable(numerical_tests_bin ${NUMERICAL_TEST_SOURCES})
target_link_libraries(numerical_tests_bin PRIVATE numerical::numerical)
# The harness include path comes from SciForge, not from a local tests/ copy.
target_include_directories(numerical_tests_bin PRIVATE "${SCIFORGE_INCLUDE_DIR}")
set_target_properties(numerical_tests_bin PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(MSVC)
    target_compile_options(numerical_tests_bin PRIVATE /W4 /WX /Zc:__cplusplus /EHsc /utf-8)
else()
    target_compile_options(numerical_tests_bin PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(NUMERICAL_SANITIZE)
    if(MSVC)
        message(FATAL_ERROR "NUMERICAL_SANITIZE requires a GCC/Clang-family compiler")
    endif()
    target_compile_options(numerical_tests_bin PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
    target_link_options(numerical_tests_bin PRIVATE -fsanitize=address,undefined)
endif()

if(NUMERICAL_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "NUMERICAL_COVERAGE requires a Clang-family compiler")
    endif()
    target_compile_options(numerical_tests_bin PRIVATE
        -fprofile-instr-generate -fcoverage-mapping -O0 -g)
    target_link_options(numerical_tests_bin PRIVATE -fprofile-instr-generate)
endif()

add_test(NAME run_tests COMMAND $<TARGET_FILE:numerical_tests_bin>)
