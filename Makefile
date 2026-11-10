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

.PHONY: headers-alone always actionlint check-pins ci-arm-gcc all build test sanitize tsan coverage coverage-build coverage-x86 coverage-html \
        lint misra doc doc-no-coverage docs-venv doc-xml docs-site docs-site-gate format format-check full-local-gate test-gcc test-gcc-neon lint-gate stubtest-gate bench-diff-selftest \
        python python-test python-sanitize python-stubtest fuzz-linalg fuzz-edges fuzz-fft fuzz-numpy fuzz-numpy-push fuzz-scipy bench-diff bench-linalg bench-python bench-fuzz bench-gemm version-check install uninstall release release-bump clean help

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
	@echo "  make python-sanitize  The Python suite under ASan + UBSan (Linux, GCC)"
	@echo "  make fuzz-linalg  Decompositions vs NumPy on adversarial matrices, every GEMM path (CASES=, SEED=)"
	@echo "  make fuzz-edges   Element-wise ops and reductions vs NumPy on zeros, infinities, NaNs, empty axes (CASES=, SEED=)"
	@echo "  make fuzz-fft     Fourier transforms vs NumPy over drawn lengths, axes, dtypes and scalings, every path (CASES=, SEED=)"
	@echo "  make fuzz-numpy   Sorting, sets, statistics, histograms, polynomials, shapes… vs NumPy on drawn inputs (CASES=, SEED=)"
	@echo "  make version-check  Assert pyproject = __init__ = CMake-derived version"
	@echo "  make check-commit  Before each commit: format, the suite under coverage at 100 % (~2 min)"
	@echo "  make check-push    Before each push: the above, g++-14, CI's arm64 g++ 13, sanitizers, MISRA, lint, doc, Python"
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
# COV_SHARDS: the processes the test run is shared among, each with its own pool (an emulator runs one thread at a
# time: see coverage-x86): one left most cores idle through the suite's serial tests (M1 Pro, eight cores: one 1371 s,
# four 923, six 689, eight 831).
COV_SHARDS ?= 6

coverage-build:
	$(CMAKE) -S . -B $(COV_DIR) -DNUMERICAL_COVERAGE=ON -DCMAKE_CXX_COMPILER=$(COV_CXX) \
	    -DSCIFORGE_INCLUDE_DIR=$(abspath $(SCIFORGE_INCLUDE))
	$(CMAKE) --build $(COV_DIR) --parallel $(JOBS)
	rm -f $(COV_DIR)/tests-*.profraw $(COV_DIR)/shard-*.log
	# One profile per process: a test's forked child counts too, and writes no file over the parent's. COV_SHARDS
	# processes share the suite (--shard=i/n, tests/main.cpp), each its output in a log of its own, printed
	# once all are done; any failing fails the run.
	@if [ $(COV_SHARDS) -le 1 ]; then \
	  echo "LLVM_PROFILE_FILE=$(COV_DIR)/tests-%p.profraw $(COV_RUN) $(COV_DIR)/numerical_tests_bin"; \
	  LLVM_PROFILE_FILE=$(COV_DIR)/tests-%p.profraw $(COV_RUN) $(COV_DIR)/numerical_tests_bin; \
	else \
	  pids=""; \
	  for i in $$(seq 0 $$(($(COV_SHARDS) - 1))); do \
	    LLVM_PROFILE_FILE=$(COV_DIR)/tests-%p.profraw \
	      $(COV_RUN) $(COV_DIR)/numerical_tests_bin --shard=$$i/$(COV_SHARDS) > $(COV_DIR)/shard-$$i.log 2>&1 & pids="$$pids $$!"; \
	  done; \
	  status=0; for pid in $$pids; do wait $$pid || status=1; done; \
	  for i in $$(seq 0 $$(($(COV_SHARDS) - 1))); do echo "── shard $$i of $(COV_SHARDS)"; cat $(COV_DIR)/shard-$$i.log; done; \
	  exit $$status; \
	fi
	$(PROFDATA) merge -sparse $(COV_DIR)/tests-*.profraw -o $(COV_DIR)/tests.profdata

coverage: coverage-build
	$(LLVM_COV) report $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    $(CURDIR)/include
	$(LLVM_COV) show $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    -format=html -output-dir=$(COV_DIR)/html -show-line-counts-or-regions
	@grep -q "NumericAL dark-coverage theme" $(COV_DIR)/html/style.css 2>/dev/null || \
	    cat docs/coverage-style.css >> $(COV_DIR)/html/style.css
	@echo "HTML coverage report: $(COV_DIR)/html/index.html"

# The x86-64 kernels (x86.hpp, fft.hpp's lanes, fft_kernels.hpp's x86 passes), which the macOS gate does not compile,
# every line and function run: on an x86-64 processor with AVX-512, or under COV_RUN (CI: Intel's emulator as a
# Skylake server), the tests take every path.
coverage-x86: coverage-build
	@$(LLVM_COV) report $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    $(CURDIR)/include/numerical/x86.hpp $(CURDIR)/include/numerical/fft.hpp $(CURDIR)/include/numerical/fft_kernels.hpp \
	    | tee $(COV_DIR)/x86.txt
	@awk '/^TOTAL/ { found = 1; if ($$6 != 0 || $$9 != 0) { print "coverage-x86: a line or function of an x86-64 kernel never ran"; \
	    exit 1 } } END { if (!found) { print "coverage-x86: no report for the x86-64 kernels"; exit 1 } }' $(COV_DIR)/x86.txt

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

# The Sphinx site (docs/site): guides, the C++ reference through Breathe from Doxygen's XML, the Python reference from
# the built extension's docstrings, with the Doxygen pages under api/. The
# toolchain is docs/requirements.txt, the pins REAL's and SciLex's sites use, installed into DOCS_VENV.
DOCS_VENV   ?= $(BUILD)/.venv-docs
SPHINXBUILD ?= $(DOCS_VENV)/bin/sphinx-build

docs-venv:
	@test -x $(SPHINXBUILD) || { python3 -m venv $(DOCS_VENV) && $(DOCS_VENV)/bin/pip install -q -r docs/requirements.txt; }

# The XML Breathe reads, from the Doxyfile with HTML off. Its own pass, quiet and not failing: the strict HTML pass
# (doc-no-coverage, WARN_AS_ERROR) reads the same sources.
doc-xml:
	@mkdir -p $(BUILD)/doc
	@( cat Doxyfile; printf 'GENERATE_HTML = NO\nGENERATE_XML = YES\nXML_OUTPUT = xml\nWARN_AS_ERROR = NO\nQUIET = YES\n' ) | doxygen - >/dev/null 2>&1
	@echo "doc-xml: $(BUILD)/doc/xml"

docs-site: docs-venv python doc-no-coverage doc-xml
	rm -rf $(BUILD)/site
	PYTHONPATH=python $(SPHINXBUILD) -q -b html -d $(BUILD)/site/doctrees docs/site $(BUILD)/site/html
	@rm -rf $(BUILD)/site/html/.doctrees $(BUILD)/site/html/api
	@cp -R $(BUILD)/doc/html $(BUILD)/site/html/api
	@! grep -rlE '/Users/|/home/runner/' $(BUILD)/site/html | head -3 | grep . \
	  || { echo "docs-site: a machine-local path is about to be published (above)"; exit 1; }
	@echo "docs-site: $(BUILD)/site/html/index.html"

# The site's gate: every Sphinx warning fatal (and all of them reported), every public Python name in the reference
# once, then every shown Python example run as a doctest. Needs no coverage build: the Doxygen XML is enough.
docs-site-gate: docs-venv python doc-no-coverage doc-xml
	rm -rf $(BUILD)/site-gate
	PYTHONPATH=python $(SPHINXBUILD) -q -W --keep-going -b html -d $(BUILD)/site-gate/doctrees docs/site $(BUILD)/site-gate/html
	@python3 tools/check_site_reference.py --self-test
	@PYTHONPATH=python python3 tools/check_site_reference.py
	@python3 tools/check_site_examples.py --self-test
	@PYTHONPATH=python python3 tools/check_site_examples.py

format:
	uncrustify -c $(SCIFORGE_LINT)/uncrustify.cfg --replace --no-backup $(FORMAT_FILES)

format-check:
	uncrustify -c $(SCIFORGE_LINT)/uncrustify.cfg --check $(FORMAT_FILES)

# The workflows' syntax, expressions and run scripts (shellcheck): a workflow that does not parse runs nothing on its
# event, so a broken release.yml publishes nothing for a tag.
actionlint:
	actionlint .github/workflows/*.yml

# Python binding (CPython Limited API, abi3 — built in place for development).
# --force: build_ext tracks the .cpp, not the headers it includes, and would relink a stale object.
python:
	SCIFORGE_INCLUDE=$(SCIFORGE_INCLUDE) $(PYTHON) setup.py -q build_ext --inplace --force

python-test: python
	$(PYRUN) -m unittest discover -s python/tests

# The binding's suite under ASan and UBSan (Linux, GCC by default), then a fixed seed's few cases of the NumPy and
# edge fuzzers: the extension and the BLAS library built instrumented in $(BUILD)/python-sanitize beside a copy of
# the package, the in-place build left as it is. The interpreter is not instrumented: the sanitizers' runtimes are
# preloaded ahead of it, and PYTHONMALLOC=malloc hands Python's own allocations to ASan. Leak detection is off:
# CPython keeps objects alive until exit.
SAN_CC  ?= gcc
SAN_CXX ?= g++
SAN_DIR := $(CURDIR)/$(BUILD)/python-sanitize
SAN_RUN  = LD_PRELOAD="$$($(SAN_CXX) -print-file-name=libasan.so) $$($(SAN_CXX) -print-file-name=libubsan.so)" \
           ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 PYTHONMALLOC=malloc \
           PYTHONPATH=$(SAN_DIR):$(abspath $(SCIFORGE_PYTHON)) $(PYTHON)
python-sanitize:
	@[ "$$(uname -s)" = Linux ] || { echo "python-sanitize: Linux only (the sanitizers' runtimes preloaded)"; exit 1; }
	rm -rf $(SAN_DIR) && mkdir -p $(SAN_DIR)
	cp -R python/numerical $(SAN_DIR)/
	find $(SAN_DIR) \( -name '*.so' -o -name '*.pyd' \) -delete
	CC=$(SAN_CC) CXX=$(SAN_CXX) NUMERICAL_SANITIZE=1 SCIFORGE_INCLUDE=$(SCIFORGE_INCLUDE) \
	  $(PYTHON) setup.py -q build_ext --build-lib $(SAN_DIR) --build-temp $(SAN_DIR)/objects --force
	@for runtime in libasan.so libubsan.so; do \
	  [ -f "$$($(SAN_CXX) -print-file-name=$$runtime)" ] || { echo "python-sanitize: $(SAN_CXX) has no $$runtime"; exit 1; }; \
	done
	$(SAN_RUN) -m unittest discover -s python/tests
	$(SAN_RUN) fuzz/npdiff.py --cases 20 --seed 20261104
	$(SAN_RUN) fuzz/edges.py --cases 1000 --seed 20261104

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

# NumPy's functions composed in Python against NumPy (fuzz/npdiff.py): CASES per family (default 1000) from SEED (default:
# the clock); fuzz-numpy-push, a fixed seed's few cases, before each push.
fuzz-numpy: python
	$(PYRUN) fuzz/npdiff.py --cases $(or $(CASES),1000) $(if $(SEED),--seed $(SEED),)

fuzz-numpy-push: python
	$(PYRUN) fuzz/npdiff.py --cases 100 --seed 20261104

# scipy.linalg's functions against SciPy (fuzz/scipy_linalg.py), then the extreme scales, each call in a process of its
# own; SciPy required.
fuzz-scipy: python
	$(PYRUN) fuzz/scipy_linalg.py --cases $(or $(CASES),600) $(if $(SEED),--seed $(SEED),)
	$(PYRUN) fuzz/scipy_linalg.py --huge

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
# CI's linux-arm job here: its compiler, Ubuntu 24.04's g++ 13 on arm64, building and running the tests as CMake does
# there (Release, CMakeLists' warnings as errors). Its -Warray-bounds and kin depend on what that version inlines at -O3:
# g++-14, and the gcc:13 image's 13.4, flagged nothing the job failed on. The image (g++, cmake, make) is built once.
CI_ARM_IMAGE ?= numerical-ci-arm-gcc:24.04
ci-arm-gcc:
	@command -v docker >/dev/null || { echo "ci-arm-gcc: docker is required (CI's linux-arm compiler runs in a container)"; exit 1; }
	@docker image inspect $(CI_ARM_IMAGE) >/dev/null 2>&1 || \
	  printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && apt-get install -y -qq --no-install-recommends g++ cmake make && rm -rf /var/lib/apt/lists/*\n' \
	  | docker build --platform linux/arm64 -q -t $(CI_ARM_IMAGE) - >/dev/null
	docker run --rm --platform linux/arm64 -v "$(CURDIR)":/src:ro -v "$(abspath $(SCIFORGE_INCLUDE))":/sciforge/include:ro $(CI_ARM_IMAGE) \
	  bash -c 'cmake -S /src -B /tmp/b -DCMAKE_BUILD_TYPE=Release -DSCIFORGE_INCLUDE_DIR=/sciforge/include > /dev/null \
	           && cmake --build /tmp/b --target numerical_tests_bin -j "$$(nproc)" 2>&1 | grep -vE "^\[ *[0-9]+%\]" ; test "$${PIPESTATUS[0]}" -eq 0 \
	           && ctest --test-dir /tmp/b --output-on-failure | tail -n 3'

test-gcc:
	@$(MAKE) --no-print-directory test CXX=g++-14 BUILD=$(BUILD)/gcc

# GCC 14 without the Apple AMX: NEON the default path, as on the aarch64 wheels and CI's linux-arm job, at the wheels'
# -O2 (GCC's vectoriser fuses a complex product into ARMv8.3's FCMLA there, which this processor has, and not at -O3).
# A test that holds on the AMX by default (and on NEON only where it asks for that path) can fail there alone.
test-gcc-neon:
	$(CMAKE) -S . -B $(BUILD)/gcc-neon -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DNUMERICAL_NO_AMX \
	    "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG"
	$(CMAKE) --build $(BUILD)/gcc-neon --parallel $(JOBS)
	$(CTEST) --test-dir $(BUILD)/gcc-neon --output-on-failure

# Each public header compiles alone, with Clang and GCC 14: a consumer may include only the one it uses, and a name a
# header takes from another it does not include compiles only behind numerical.hpp's order.
HEADERS_ALONE_CXX ?= clang++ g++-14
ALONE_HEADERS     := $(notdir $(wildcard include/numerical/*.hpp))
headers-alone:
	@$(MAKE) --no-print-directory -j$(JOBS) $(ALONE_HEADERS:%=alone/%)
	@echo "headers-alone: each of the $(words $(ALONE_HEADERS)) headers compiles alone"

alone/%: always
	@for c in $(HEADERS_ALONE_CXX); do \
	  out=$$(echo '#include <numerical/$*>' | $$c -std=c++20 -fsyntax-only -Iinclude -I"$(SCIFORGE_INCLUDE)" -x c++ - 2>&1) \
	    || { echo "$$out"; echo "headers-alone: $* does not compile alone ($$c)"; exit 1; }; \
	done

always:

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
# process of its own; the distribution of the ratios and the worst cases (CASES=, SEED=, LARGEST=). FAMILY=scipy draws
# scipy.linalg's functions against SciPy instead, a population of its own.
# Defaults of its own, as fuzz-linalg keeps its own for CASES and SEED. EIGEN= (or EIGEN_INCLUDE=), Eigen's include
# directory, adds Eigen against NumericAL from C++ (benchmarks/fuzz_worker.cpp); EIGEN_FLAGS= adds to Eigen's worker's
# flags, EIGEN_OPENMP=1 builds it with OpenMP, NUMERICAL_THREADS= caps NumericAL's C++ worker's threads.
EIGEN_DIR := $(or $(EIGEN),$(EIGEN_INCLUDE))
bench-fuzz: python
	$(PYRUN) benchmarks/bench_fuzz.py --cases $(or $(CASES),300) --seed $(or $(SEED),20261021) $(if $(FAMILY),--family $(FAMILY)) \
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
# Cheapest first, so a slip in a comment or a stub fails in seconds, not after the suites: Doxygen before coverage, the
# static checks (MISRA, the stubs, lint) before the builds and runs.
QUALITY_COMMIT   := format-check actionlint headers-alone doc-no-coverage coverage-gate
QUALITY_PUSH     := docs-site-gate version-check check-pins misra stubtest-gate lint-gate bench-diff-selftest python-test fuzz-numpy-push test test-gcc test-gcc-neon \
                    ci-arm-gcc sanitize tsan
QUALITY_RELEASE  := bench-diff fuzz-linalg fuzz-edges fuzz-fft fuzz-numpy fuzz-scipy
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
