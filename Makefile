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
FORMAT_FILES := $(shell find include tests -name '*.hpp' -o -name '*.cpp')

.PHONY: check-pins all build test sanitize coverage coverage-build coverage-html \
        lint misra doc doc-no-coverage format format-check full-local-gate \
        python python-test fuzz-linalg bench-linalg bench-python bench-gemm version-check install uninstall release clean help

.DEFAULT_GOAL := help

help:
	@echo "NumericAL — build orchestrator (CMake + QA tools)"
	@echo ""
	@echo "  make build      Configure and build the test binary (CMake)"
	@echo "  make test       Build and run the test suite (ctest)"
	@echo "  make sanitize   Build and run the tests under ASan + UBSan"
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
	@echo "  make version-check  Assert pyproject = __init__ = CMake-derived version"
	@echo "  make full-local-gate  Every gate in one command (the macOS gate of record)"
	@echo "  make bench-python Differential validation + perf vs NumPy"
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

coverage-build:
	$(CMAKE) -S . -B $(COV_DIR) -DNUMERICAL_COVERAGE=ON -DCMAKE_CXX_COMPILER=$(COV_CXX)
	$(CMAKE) --build $(COV_DIR) --parallel $(JOBS)
	rm -f $(COV_DIR)/tests-*.profraw
	# One profile per process: a test's forked child counts too, and writes no file over the parent's.
	LLVM_PROFILE_FILE=$(COV_DIR)/tests-%p.profraw $(COV_DIR)/numerical_tests_bin
	$(PROFDATA) merge -sparse $(COV_DIR)/tests-*.profraw -o $(COV_DIR)/tests.profdata

coverage: coverage-build
	$(LLVM_COV) report $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    $(CURDIR)/include
	$(LLVM_COV) show $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    -format=html -output-dir=$(COV_DIR)/html -show-line-counts-or-regions
	@grep -q "NumericAL dark-coverage theme" $(COV_DIR)/html/style.css 2>/dev/null || \
	    cat docs/coverage-style.css >> $(COV_DIR)/html/style.css
	@echo "HTML coverage report: $(COV_DIR)/html/index.html"

coverage-html:
	@mkdir -p $(COV_DIR)
	@$(MAKE) --silent coverage-build > $(COV_DIR)/build.log 2>&1 || (cat $(COV_DIR)/build.log; exit 1)
	@$(LLVM_COV) show $(COV_DIR)/numerical_tests_bin -instr-profile=$(COV_DIR)/tests.profdata \
	    -format=html -output-dir=$(COV_DIR)/html -show-line-counts-or-regions
	@grep -q "NumericAL dark-coverage theme" $(COV_DIR)/html/style.css 2>/dev/null || \
	    cat docs/coverage-style.css >> $(COV_DIR)/html/style.css
	@echo "HTML coverage report: $(COV_DIR)/html/index.html"

lint:
	@ls tests/*.cpp | xargs -P $(JOBS) -I{} clang-tidy {} -- $(CXXSTD) $(INCLUDES) -I$(SCIFORGE_INCLUDE)

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

# A campaign: CASES matrices per path (default 500), from SEED (default: the clock, printed for reproduction).
fuzz-linalg: python
	$(PYRUN) fuzz/linalg.py --cases $(or $(CASES),500) $(if $(SEED),--seed $(SEED),)

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

# The complete local quality gate in one command — the canonical pre-push check
# and, since CI no longer runs macOS (see .github/workflows/ci.yml), the macOS gate
# of record. Runs every check this machine owns and fails on any issue, including a
# lint warning or any coverage dimension below 100%. If an incremental coverage build
# is ever suspect, `rm -rf $(COV_DIR)` first to force a clean re-measure.
# Pin-drift lint: fail if this repo pins more than one SciForge version or a sciforge-build floor above the
# pin (SciForge's tools/check-pins.sh). Skipped with a warning when the sibling tool is absent.
SCIFORGE_TOOLS ?= ../sciforge/tools
check-pins:
	@if test -x $(SCIFORGE_TOOLS)/check-pins.sh; then $(SCIFORGE_TOOLS)/check-pins.sh .; \
	 else echo "check-pins: WARN — $(SCIFORGE_TOOLS)/check-pins.sh absent, skipped"; fi

full-local-gate:
	@$(MAKE) format-check
	@$(MAKE) version-check
	@$(MAKE) check-pins
	@$(MAKE) test
	@$(MAKE) test CXX=g++-14 BUILD=$(BUILD)/gcc
	@$(MAKE) sanitize
	@$(MAKE) misra
	@$(MAKE) doc-no-coverage
	@$(MAKE) python-test
	@$(MAKE) lint | tee $(BUILD)/lint.log; ! grep -qE 'warning:|error:' $(BUILD)/lint.log
	@$(MAKE) coverage | tee $(BUILD)/coverage.log
	# Gate on a flag, not a bare exit 1 in the rule: that exit is overridden by END's exit,
	# so the gate silently accepted coverage below 100% 4D before this fix.
	@awk '/^TOTAL/{seen=1; if (gsub(/100\.00%/, "&") != 4) bad=1} END{exit (seen && !bad) ? 0 : 1}' $(BUILD)/coverage.log \
	  || { echo "full-local-gate: coverage is below 100% on some dimension — see above"; exit 1; }
	@echo "full-local-gate: ALL gates green (clang + g++-14, sanitize, MISRA, lint, doc, python, version-check, 100% coverage)"

# Differential validation + performance versus NumPy (needs numpy installed).
bench-python: python
	$(PYRUN) benchmarks/bench.py

bench-linalg: python
	$(PYRUN) benchmarks/linalg_bench.py --sizes $(or $(SIZES),64,128,256,512,1024) --rounds $(or $(ROUNDS),3) \
	    --json $(BUILD)/bench-linalg.json

# Standalone C++ GEMM micro-benchmark: the shipping operator* against a naive
# column-strided ijk baseline and a blocked variant (see BENCHMARKS.md). Built
# with optimisation, outside CMake; not part of the gate.
bench-gemm:
	@mkdir -p $(BUILD)
	$(CXX) $(CXXSTD) -O2 $(INCLUDES) -I$(SCIFORGE_INCLUDE) -pthread benchmarks/gemm_bench.cpp -o $(BUILD)/gemm_bench
	$(PYRUN) benchmarks/gemm_bench.py $(BUILD)/gemm_bench

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
release:
	@test "$$(git symbolic-ref --short HEAD)" = main || { echo "release from main only"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "working tree not clean"; exit 1; }
	@git fetch --tags --quiet origin
	@year=$$(date -u +%Y); month=$$(date -u +%m | sed 's/^0//'); today=$$(date -u +%Y-%m-%d); \
	 patch=$$(git tag -l "v$$year.$$month.*" | wc -l | tr -d ' '); \
	 version="$$year.$$month.$$patch"; \
	 echo "Releasing v$$version ($$today)$${DRY_RUN:+  [DRY_RUN — no commit/tag/push]}"; \
	 awk '/^## Unreleased$$/{f=1;next} f&&/^## /{exit 1} f&&NF{found=1;exit 0} END{if(!found)exit 1}' CHANGELOG.md \
	   || { echo "release: '## Unreleased' is empty — add a CHANGELOG line (even 'maintenance release') first"; exit 1; }; \
	 tb=$$(mktemp); \
	 awk -v v="$$version" -v d="$$today" '/^## Unreleased$$/{print; print ""; print "## " v " — " d; next} {print}' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md; \
	 awk -v h="## $$version " 'index($$0,h)==1{f=1;print;next} f&&/^## /{exit} f{print}' CHANGELOG.md > $$tb; \
	 sed -i.bak -E "s/^version = \".*\"/version = \"$$version\"/" pyproject.toml && rm -f pyproject.toml.bak; \
	 sed -i.bak -E "s/^__version__ = \".*\"/__version__ = \"$$version\"/" python/numerical/__init__.py && rm -f python/numerical/__init__.py.bak; \
	 if [ -n "$$DRY_RUN" ]; then \
	   echo "would commit 'release: v$$version' (pyproject, __init__, CHANGELOG); annotated tag v$$version, body:"; \
	   sed 's/^/    | /' $$tb; \
	   echo "would run: git push origin HEAD v$$version"; \
	   git checkout -- pyproject.toml python/numerical/__init__.py CHANGELOG.md; \
	 else \
	   git add pyproject.toml python/numerical/__init__.py CHANGELOG.md; \
	   git commit -m "release: v$$version"; \
	   git tag -a --cleanup=verbatim "v$$version" -F $$tb; \
	   git push origin HEAD "v$$version"; \
	 fi; \
	 rm -f $$tb

clean:
	rm -rf $(BUILD) build dist ./*.egg-info python/numerical/*.so
	find python -name '__pycache__' -type d -prune -exec rm -rf {} +
