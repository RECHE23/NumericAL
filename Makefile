# Thin orchestrator over CMake (build/test/sanitize/coverage) and the QA tools
# (clang-tidy, doxygen, uncrustify). CMake owns all compilation policy
# (CMakeLists.txt); this file only wires the frequent commands.
#
# Override the compiler with CXX on the command line, e.g.
#   make test CXX=g++-14
# Switching compilers reuses a cached build dir, so run `make clean` first.
#
# NumericAL is a header-only leaf library: no dependencies beyond the C++ standard
# library.

CMAKE  ?= cmake
CTEST  ?= ctest
PYTHON ?= python3
# SciForge owns the shared, dev-only benchmark substrate (sciforge.bench: dep-free stats,
# schema, collector), in its python/ dir. Same sibling default as SCIFORGE_INCLUDE/_LINT.
# Never a build/runtime dependency — the bench imports it only when run, never shipped.
SCIFORGE_PYTHON ?= ../sciforge/python
# Append the SciForge sibling so benchmarks can `import sciforge.bench` (shared stats).
PYRUN  := PYTHONPATH=$(CURDIR)/python:$(abspath $(SCIFORGE_PYTHON)) $(PYTHON)
BUILD  := build
# Parallelism: detected core count (override with JOBS=N).
JOBS   ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

ifeq ($(origin CXX),command line)
CMAKE_CXX := -DCMAKE_CXX_COMPILER=$(CXX)
endif

CXXSTD       := -std=c++20
INCLUDES     := -Iinclude
# The test harness (framework.hpp) is owned by SciForge; the test TUs include
# it as <sciforge/test/framework.hpp>. clang-tidy (make lint) needs that path too.
SCIFORGE_INCLUDE ?= ../sciforge/include
# SciForge also owns the shared lint config (the MISRA base + uncrustify.cfg), in
# its lint/ dir. Same sibling default; CI checks SciForge out alongside as ../sciforge.
SCIFORGE_LINT ?= ../sciforge/lint
FORMAT_FILES := $(shell find include tests blas -name '*.hpp' -o -name '*.cpp')

.PHONY: check-pins all build test sanitize tsan coverage coverage-build coverage-x86 coverage-html \
        lint misra doc doc-no-coverage format format-check full-local-gate test-gcc test-gcc-neon lint-gate stubtest-gate bench-diff-selftest \
        python python-test python-stubtest fuzz-linalg fuzz-edges fuzz-fft bench-diff bench-linalg bench-python bench-fuzz bench-gemm version-check install uninstall release release-bump clean help

.DEFAULT_GOAL := help

help:
	@echo "NumericAL — build orchestrator (CMake + QA tools)"
	@echo ""
	@echo "  make build      Configure and build the test binary (CMake)"
	@echo "  make test       Build and run the test suite (ctest)"
	@echo "  make sanitize   Build and run the tests under ASan + UBSan"
	@echo "  make tsan       Build and run the tests under ThreadSanitizer (a planted race reported first)"
	@echo "  make test-gcc-neon  The tests under GCC 14 without the AMX: NEON the default path, as on the aarch64 wheels"
	@echo "  make coverage   Line-coverage text summary + HTML report"
	@echo "  make lint       clang-tidy over the test sources"
	@echo "  make misra      MISRA C++:2023-oriented analysis"
	@echo "  make doc        Generate API reference (Doxygen) with embedded coverage"
	@echo "  make doc-no-coverage  Generate API reference without coverage report"
	@echo "  make format     Uncrustify, in place"
	@echo "  make format-check  Uncrustify, dry-run, exits non-zero on diff"
	@echo "  make python     Build the Python extension in place (abi3)"
	@echo "  make python-test  Run the Python binding test suite"
	@echo "  make fuzz-linalg  Decompositions vs NumPy on adversarial matrices, every GEMM path (CASES=, SEED=)"
	@echo "  make fuzz-edges   Element-wise ops and reductions vs NumPy on zeros, infinities, NaNs, empty axes (CASES=, SEED=)"
	@echo "  make fuzz-fft     Fourier transforms vs NumPy over drawn lengths, axes, dtypes and scalings, every path (CASES=, SEED=)"
	@echo "  make version-check  Assert pyproject = __init__ = CMake-derived version"
	@echo "  make check-commit  Before each commit: format, the suite under coverage at 100 % (~2 min)"
	@echo "  make check-push    Before each push: the above, g++-14, sanitizers, MISRA, lint, doc, Python (~6 min)"
	@echo "  make check-release Before each release: the above, bench-diff against the last release, the fuzzers"
	@echo "  make full-local-gate  The push tier, by its former name"
	@echo "  make bench-python Differential validation + perf vs NumPy"
	@echo "  make bench-fuzz   Drawn cases vs NumPy: the ratios' distribution and the worst (CASES=, SEED=, LARGEST=,"
	@echo "                    EIGEN= adds Eigen against NumericAL from C++)"
	@echo "  make bench-linalg Dense linear algebra vs NumPy's BLAS/LAPACK, each alone in its process (SIZES=, ROUNDS=)"
	@echo "  make bench-gemm   GEMM loop-order micro-benchmark (C++, vs naive baseline)"
	@echo "  make release    Cut a calendar-versioned release (tag + push)"
	@echo "  make clean      Remove build artifacts"
	@echo ""
	@echo "  Override the compiler: make test CXX=g++-14"

all: build

build:
	$(CMAKE) -S . -B $(BUILD) $(CMAKE_CXX) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(BUILD) --parallel $(JOBS)

test: build
	$(CTEST) --test-dir $(BUILD) --output-on-failure

sanitize:
	$(CMAKE) -S . -B $(BUILD)/sanitize $(CMAKE_CXX) -DNUMERICAL_SANITIZE=ON
	$(CMAKE) --build $(BUILD)/sanitize --parallel $(JOBS)
	$(CTEST) --test-dir $(BUILD)/sanitize --output-on-failure

# ThreadSanitizer over the suite: the pool's workers, nested parallel_for, the buffers each thread keeps. A race
# planted in a parallel_for (tests/tsan/planted_race.cpp) must be reported first, else a clean run proves nothing.
# TSan cannot map its shadow memory under the high-entropy ASLR of recent Linux kernels: run without it there.
ifeq ($(shell uname -s),Darwin)
TSAN_CXX ?= /usr/bin/clang++
else
TSAN_CXX ?= clang++
endif
TSAN_RUN := $(if $(shell command -v setarch 2>/dev/null),setarch $(shell uname -m) -R,)
tsan:
	@mkdir -p $(BUILD)/tsan
	$(TSAN_CXX) $(CXXSTD) -O1 -g -fsanitize=thread $(INCLUDES) tests/tsan/planted_race.cpp -o $(BUILD)/tsan/planted_race
	@if $(TSAN_RUN) $(BUILD)/tsan/planted_race > $(BUILD)/tsan/planted_race.log 2>&1 || true; \
	   ! grep -q "WARNING: ThreadSanitizer: data race" $(BUILD)/tsan/planted_race.log; then \
	   cat $(BUILD)/tsan/planted_race.log; echo "tsan: the planted race was not reported"; exit 1; fi
	$(CMAKE) -S . -B $(BUILD)/tsan/suite -DCMAKE_CXX_COMPILER=$(TSAN_CXX) -DNUMERICAL_TSAN=ON
	$(CMAKE) --build $(BUILD)/tsan/suite --parallel $(JOBS)
	$(TSAN_RUN) $(BUILD)/tsan/suite/numerical_tests_bin

# Coverage uses LLVM source-based instrumentation, so it pins a Clang toolchain
# end to end. On macOS the Apple toolchain is required (Homebrew clang links a
# profile runtime its llvm-profdata cannot read).
ifeq ($(shell uname -s),Darwin)
COV_CXX  ?= /usr/bin/clang++
PROFDATA ?= xcrun llvm-profdata
LLVM_COV ?= xcrun llvm-cov
else
COV_CXX  ?= clang++
PROFDATA ?= llvm-profdata
LLVM_COV ?= llvm-cov
endif
COV_DIR  := $(BUILD)/coverage

# COV_RUN prefixes the test run (an emulator: see coverage-x86).
COV_RUN ?=

coverage-build:
	$(CMAKE) -S . -B $(COV_DIR) -DNUMERICAL_COVERAGE=ON -DCMAKE_CXX_COMPILER=$(COV_CXX) \
	    -DSCIFORGE_INCLUDE_DIR=$(abspath $(SCIFORGE_INCLUDE))
	$(CMAKE) --build $(COV_DIR) --parallel $(JOBS)
	rm -f $(COV_DIR)/tests-*.profraw
	# One profile per process: a test's forked child counts too, and writes no file over the parent's.
	LLVM_PROFILE_FILE=$(COV_DIR)/tests-%p.profraw $(COV_RUN) $(COV_DIR)/numerical_tests_bin
	$(PROFDATA) merge -sparse $(COV_DIR)/tests-*.profraw -o $(COV_DIR)/tests.profdata

coverage: coverage-build
	$(LLVM_COV) report $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    $(CURDIR)/include
	$(LLVM_COV) show $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    -format=html -output-dir=$(COV_DIR)/html -show-line-counts-or-regions
	@grep -q "NumericAL dark-coverage theme" $(COV_DIR)/html/style.css 2>/dev/null || \
	    cat docs/coverage-style.css >> $(COV_DIR)/html/style.css
	@echo "HTML coverage report: $(COV_DIR)/html/index.html"

# The x86-64 kernels (x86.hpp), which the macOS gate does not compile, every line and function run: on an x86-64
# processor with AVX-512, or under COV_RUN (CI: Intel's emulator as a Skylake server), the tests take every path.
coverage-x86: coverage-build
	@$(LLVM_COV) report $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    $(CURDIR)/include/numerical/x86.hpp | tee $(COV_DIR)/x86.txt
	@awk '/^TOTAL/ { found = 1; if ($$6 != 0 || $$9 != 0) { print "coverage-x86: a line or function of x86.hpp never ran"; \
	    exit 1 } } END { if (!found) { print "coverage-x86: no report for x86.hpp"; exit 1 } }' $(COV_DIR)/x86.txt

coverage-html:
	@mkdir -p $(COV_DIR)
	@$(MAKE) --silent coverage-build > $(COV_DIR)/build.log 2>&1 || (cat $(COV_DIR)/build.log; exit 1)
	@$(LLVM_COV) show $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    -format=html -output-dir=$(COV_DIR)/html -show-line-counts-or-regions
	@grep -q "NumericAL dark-coverage theme" $(COV_DIR)/html/style.css 2>/dev/null || \
	    cat docs/coverage-style.css >> $(COV_DIR)/html/style.css
	@echo "HTML coverage report: $(COV_DIR)/html/index.html"

lint:
	@ls tests/*.cpp blas/*.cpp | xargs -P $(JOBS) -I{} clang-tidy {} -- $(CXXSTD) $(INCLUDES) -I$(SCIFORGE_INCLUDE)

# The clang-tidy CI pins, named in SciForge's lint/misra.mk (MISRA_TIDY, MISRA_TIDY_CHECK).
include $(SCIFORGE_LINT)/misra.mk
misra:
	mkdir -p $(BUILD)
	printf '#include <numerical/numerical.hpp>\nint main(){ try { const numerical::vector<int> v {1, 2, 3}; return numerical::dot(v, v) == 14 ? 0 : 1; } catch (...) { return 2; } }\n' > $(BUILD)/misra_tu.cpp
	@$(MISRA_TIDY_CHECK)
	$(MISRA_TIDY) --config-file=$(SCIFORGE_LINT)/clang-tidy-misra \
	    --header-filter='include/numerical/.*' \
	    $(BUILD)/misra_tu.cpp -- $(CXXSTD) $(INCLUDES)

doc: coverage-html
	mkdir -p $(BUILD)/doc
	doxygen Doxyfile
	@rm -rf $(BUILD)/doc/html/coverage
	@cp -R $(COV_DIR)/html $(BUILD)/doc/html/coverage
	@echo "API reference: $(BUILD)/doc/html/index.html"

doc-no-coverage:
	mkdir -p $(BUILD)/doc
	doxygen Doxyfile
	@echo "API reference: $(BUILD)/doc/html/index.html"

format:
	uncrustify -c $(SCIFORGE_LINT)/uncrustify.cfg --replace --no-backup $(FORMAT_FILES)

format-check:
	uncrustify -c $(SCIFORGE_LINT)/uncrustify.cfg --check $(FORMAT_FILES)

# Python binding (CPython Limited API, abi3 — built in place for development).
# --force: build_ext tracks the .cpp, not the headers it includes, and would relink a stale object.
python:
	SCIFORGE_INCLUDE=$(SCIFORGE_INCLUDE) $(PYTHON) setup.py -q build_ext --inplace --force

python-test: python
	$(PYRUN) -m unittest discover -s python/tests

# The stubs are checked against the runtime surface (mypy.stubtest): a function or Tensor method added to the
# binding or the package and not to numerical/__init__.pyi or _numerical.pyi fails here. MYPY_PYTHON carries mypy.
MYPY_PYTHON ?= $(PYTHON)
python-stubtest: python
	PYTHONPATH=python $(MYPY_PYTHON) -m mypy.stubtest numerical

# A campaign: CASES matrices per path (default 500), from SEED (default: the clock, printed for reproduction).
fuzz-linalg: python
	$(PYRUN) fuzz/linalg.py --cases $(or $(CASES),500) $(if $(SEED),--seed $(SEED),)

fuzz-edges: python
	$(PYRUN) fuzz/edges.py --cases $(or $(CASES),20000) $(if $(SEED),--seed $(SEED),)

fuzz-fft: python
	$(PYRUN) fuzz/fft.py --cases $(or $(CASES),300) $(if $(SEED),--seed $(SEED),)

# Version-consistency gate: pyproject.toml is the single source of truth — `make
# release` bumps __init__.py from it, CMakeLists.txt derives it. Asserts the three
# agree and that CMake still DERIVES (no hardcoded literal that could drift; the
# release template never seds CMakeLists, so a literal there silently goes stale).
version-check:
	@py=$$(sed -nE 's/^version = "([0-9][0-9.]*)"/\1/p' pyproject.toml); \
	 ini=$$(sed -nE 's/^__version__ = "([0-9][0-9.]*)"/\1/p' python/numerical/__init__.py); \
	 lit=$$(sed -nE 's/^project\([A-Za-z_]+ VERSION ([0-9][0-9.]*).*/\1/p' CMakeLists.txt); \
	 if [ -z "$$py" ]; then echo "version-check: no version found in pyproject.toml"; exit 1; fi; \
	 if [ "$$py" != "$$ini" ]; then echo "version-check: DRIFT pyproject=$$py vs __init__=$$ini"; exit 1; fi; \
	 if [ -n "$$lit" ]; then echo "version-check: CMakeLists.txt hardcodes VERSION $$lit (must derive from pyproject.toml = $$py)"; exit 1; fi; \
	 if ! grep -q 'file(READ.*pyproject\.toml' CMakeLists.txt; then echo "version-check: CMakeLists.txt must derive its version from pyproject.toml"; exit 1; fi; \
	 echo "version-check: $$py (pyproject = __init__ = CMake-derived)"

# Pin-drift lint: fail if this repo pins more than one SciForge version or a sciforge-build floor above the
# pin (SciForge's tools/check-pins.sh). Skipped with a warning when the sibling tool is absent.
SCIFORGE_TOOLS ?= ../sciforge/tools
check-pins:
	@if test -x $(SCIFORGE_TOOLS)/check-pins.sh; then $(SCIFORGE_TOOLS)/check-pins.sh .; \
	 else echo "check-pins: WARN — $(SCIFORGE_TOOLS)/check-pins.sh absent, skipped"; fi

# The complete local quality gate in one command — the canonical pre-push check
# and, since CI no longer runs macOS (see .github/workflows/ci.yml), the macOS gate
# of record. Runs every check this machine owns and fails on any issue, including a
# lint warning or any coverage dimension below 100%. If an incremental coverage build
# is ever suspect, `rm -rf $(COV_DIR)` first to force a clean re-measure.
# The steps of the verification tiers (make/quality.mk, below) that take arguments, as targets of their own.
test-gcc:
	@$(MAKE) --no-print-directory test CXX=g++-14 BUILD=$(BUILD)/gcc

# GCC 14 without the Apple AMX: NEON the default path, as on the aarch64 wheels and CI's linux-arm job. A test that
# holds on the AMX by default (and on NEON only where it asks for that path) can fail there alone.
test-gcc-neon:
	$(CMAKE) -S . -B $(BUILD)/gcc-neon -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DNUMERICAL_NO_AMX
	$(CMAKE) --build $(BUILD)/gcc-neon --parallel $(JOBS)
	$(CTEST) --test-dir $(BUILD)/gcc-neon --output-on-failure

lint-gate:
	@$(MAKE) --no-print-directory lint | tee $(BUILD)/lint.log; ! grep -qE 'warning:|error:' $(BUILD)/lint.log

stubtest-gate:
	@if $(MYPY_PYTHON) -c 'import mypy' >/dev/null 2>&1; then $(MAKE) --no-print-directory python-stubtest; \
	 else echo "stubtest-gate: WARN — mypy absent from MYPY_PYTHON, python-stubtest skipped (CI runs it)"; fi

bench-diff-selftest:
	@$(PYTHON) benchmarks/bench_diff.py --self-test

full-local-gate:
	@$(MAKE) --no-print-directory check-push

# Differential validation + performance versus NumPy (needs numpy installed).
bench-python: python
	$(PYRUN) benchmarks/bench.py

# Drawn cases versus NumPy: operations, shapes on a log scale, dtypes and operand layouts, each library timed in a
# process of its own; the distribution of the ratios and the worst cases (CASES=, SEED=, LARGEST=).
# Defaults of its own, as fuzz-linalg keeps its own for CASES and SEED. EIGEN= (or EIGEN_INCLUDE=), Eigen's include
# directory, adds Eigen against NumericAL from C++ (benchmarks/fuzz_worker.cpp); EIGEN_FLAGS= adds to Eigen's worker's
# flags, EIGEN_OPENMP=1 builds it with OpenMP, NUMERICAL_THREADS= caps NumericAL's C++ worker's threads.
EIGEN_DIR := $(or $(EIGEN),$(EIGEN_INCLUDE))
bench-fuzz: python
	$(PYRUN) benchmarks/bench_fuzz.py --cases $(or $(CASES),300) --seed $(or $(SEED),20261021) \
	    --largest $(or $(LARGEST),1024) --json $(BUILD)/bench-fuzz.json \
	    $(if $(EIGEN_DIR),--eigen '$(EIGEN_DIR)') $(if $(EIGEN_FLAGS),'--eigen-flags=$(EIGEN_FLAGS)') \
	    $(if $(EIGEN_OPENMP),--eigen-openmp) $(if $(NUMERICAL_THREADS),--numerical-threads $(NUMERICAL_THREADS))

# The default sizes in a variable: inside $(or …) its commas would split the arguments, leaving 64 alone.
SIZES  ?= 64,128,256,512,1024
ROUNDS ?= 3

bench-linalg: python
	$(PYRUN) benchmarks/linalg_bench.py --sizes $(SIZES) --rounds $(ROUNDS) $(if $(NUMPY_INPUTS),--numpy-inputs) \
	    --json $(BUILD)/bench-linalg.json

# Standalone C++ GEMM micro-benchmark: the shipping operator* against a naive
# column-strided ijk baseline and a blocked variant (see BENCHMARKS.md). Built
# with optimisation, outside CMake; not part of the gate.
bench-gemm:
	@mkdir -p $(BUILD)
	$(CXX) $(CXXSTD) -O2 $(INCLUDES) -I$(SCIFORGE_INCLUDE) -pthread benchmarks/gemm_bench.cpp -o $(BUILD)/gemm_bench
	$(PYRUN) benchmarks/gemm_bench.py $(BUILD)/gemm_bench

# The decompositions of the last release against the working tree, built with each compiler the wheels use, at
# the orders between the powers of two too, alternately; a cell slower by 5 % in two batches fails it. BASE= names
# another release, HEAD_REF= one to stand for the working tree. `make release` runs it first (SKIP_BENCH_DIFF=1
# skips it, knowingly).
bench-diff:
	$(PYTHON) benchmarks/bench_diff.py --self-test
	$(PYTHON) benchmarks/bench_diff.py $(if $(BASE),--base $(BASE),) $(if $(HEAD_REF),--head $(HEAD_REF),)

install:
	$(PYTHON) -m pip install .

uninstall:
	$(PYTHON) -m pip uninstall -y numerical

# Cuts a calendar-versioned release: computes YYYY.M.PATCH with the patch reset
# each month (first release of a month is .0; PEP 440 drops leading zeros, so
# 2026.6.1, never 2026.06.001), bumps pyproject.toml + __init__.py (CMakeLists.txt
# derives its version from pyproject.toml, so it follows automatically), commits, tags and
# pushes from a clean main. Pushing the tag drives release.yml, which checks the
# tag matches the version, builds the wheels and sdist, and publishes to PyPI via
# Trusted Publishing (OIDC) — once the one-time Trusted Publisher is configured
# (see README). The pushed tag is the single thing that triggers a publish.
#
# The CHANGELOG's Unreleased section becomes the version's, and the body of an ANNOTATED tag
# (--cleanup=verbatim keeps its "## " headings, which git would drop as comments); release.yml publishes
# that body as the GitHub release notes. An empty Unreleased section refuses the release. DRY_RUN=1 runs
# the computation and that guard with no commit, tag or push.
# The version `make release` (SciForge's make/release.mk, below) writes: pyproject.toml is its source, CMake derives
# it, __init__.py carries it for Python.
release-bump:
	@test -n "$(VERSION)" || { echo "release-bump: VERSION= is required"; exit 1; }
	@sed -i.bak -E "s/^version = \".*\"/version = \"$(VERSION)\"/" pyproject.toml && rm -f pyproject.toml.bak
	@sed -i.bak -E "s/^__version__ = \".*\"/__version__ = \"$(VERSION)\"/" python/numerical/__init__.py && rm -f python/numerical/__init__.py.bak

clean:
	rm -rf $(BUILD) build dist ./*.egg-info python/numerical/*.so python/numerical/lib
	find python -name '__pycache__' -type d -prune -exec rm -rf {} +

# The verification tiers every SciForge project shares (the sibling checkout's make/quality.mk). Optional, so a
# checkout without the sibling (CI pins its own SciForge) still runs every other target.
SCIFORGE_MAKE    ?= ../sciforge/make
QUALITY_COMMIT   := format-check coverage-gate
QUALITY_PUSH     := version-check check-pins test test-gcc test-gcc-neon sanitize tsan misra doc-no-coverage python-test bench-diff-selftest \
                    stubtest-gate lint-gate
QUALITY_RELEASE  := bench-diff fuzz-linalg fuzz-edges fuzz-fft
-include $(SCIFORGE_MAKE)/quality.mk

# The calendar release every SciForge project cuts (the sibling checkout's make/release.mk): bench-diff against the last
# release first (SKIP_BENCH_DIFF=1, or SKIP_PRECHECK=1, to release anyway), CI green on the commit tagged.
RELEASE_FILES    := pyproject.toml python/numerical/__init__.py
# ci.yml's paths-ignore: a HEAD changed in those paths alone is judged by ci.yml's run before it.
# The AVX-512 emulator's run, which pushes outside the x86-64 kernels skip, must also have passed on the commit tagged.
RELEASE_CI       := ci.yml avx512.yml
RELEASE_CI_IGNORED := *.md docs/*
RELEASE_PRECHECK := $(if $(SKIP_BENCH_DIFF),,bench-diff)
-include $(SCIFORGE_MAKE)/release.mk
