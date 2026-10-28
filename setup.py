# Builds the abi3 C++ extension for the NumericAL Python binding. Run from the
# repository root (the sdist/wheel build entry); metadata lives in pyproject.toml.
# NumericAL is header-only (headers in include/) and a leaf — it depends on nothing
# beyond the C++ standard library.
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.errors import CompileError

try:  # setuptools >= 70.1 vendors bdist_wheel; older installs get it from wheel
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover
    from wheel.bdist_wheel import bdist_wheel


def _sciforge_include():
    """Locate SciForge's binding headers: an explicit SCIFORGE_INCLUDE override, then a
    sibling checkout (local dev wins over a possibly stale pip-installed package), then the
    sciforge-build package."""
    header = os.path.join("sciforge", "binding", "error.hpp")
    env = os.environ.get("SCIFORGE_INCLUDE")
    if env and os.path.exists(os.path.join(env, header)):
        return env
    sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sciforge", "include")
    if os.path.exists(os.path.join(sibling, header)):
        return sibling
    try:
        import sciforge_build
        return sciforge_build.get_include()
    except ImportError:
        raise SystemExit("sciforge headers introuvables : checkout RECHE23/sciforge en sibling, "
                         "ou pip install sciforge-build, ou définir SCIFORGE_INCLUDE")


HERE = os.path.dirname(os.path.abspath(__file__))


class build_py_with_headers(build_py):
    """Ships the header-only C++ library inside the package so an installed
    wheel can be located by numerical.get_include() for C++ integration."""

    def run(self):
        super().run()
        src = os.path.join(HERE, "include", "numerical")
        dst = os.path.join(self.build_lib, "numerical", "include", "numerical")
        os.makedirs(dst, exist_ok=True)
        for name in sorted(os.listdir(src)):
            if name.endswith(".hpp"):
                shutil.copy2(os.path.join(src, name), os.path.join(dst, name))


class abi3_wheel(bdist_wheel):
    """Forces the stable-ABI tag so one cp310-abi3 wheel serves CPython 3.10+."""

    def finalize_options(self):
        super().finalize_options()
        self.py_limited_api = "cp310"


if sys.platform == "win32":
    compile_args = ["/std:c++20", "/O2", "/EHsc", "/Zc:__cplusplus"]
    link_args = []
else:
    # -g0 after the interpreter's own -g: a wheel carries no debug information. -O3 on x86-64, where GCC's
    # unrolling and vectorising took 7-18 % off the decompositions of order 64 to 512 (solve, inverse,
    # cholesky, qr, eigh, svd); on arm64 it moved some up and some down, so -O2 stays there.
    level = "-O3" if platform.machine().lower() in ("x86_64", "amd64") else "-O2"
    # -ffp-contract=on (Clang's default) fuses a multiply-add within one expression only. GCC's default, fast, fuses
    # across statements as its passes find them, so a loop's vectorised version and its scalar one (which runs,
    # chosen at run time by the operands' addresses) could round the same element differently: an eigvalsh of
    # order 6 answered with other bits after other allocations (AVX-512, GCC 15 -O3).
    compile_args = ["-std=c++20", level, "-g0", "-fvisibility=hidden", "-pthread", "-ffp-contract=on"]
    link_args = ["-pthread"]  # parallel.hpp uses std::thread

# Clang's loop vectoriser takes a loop that sums products (`s += a[i] * b[i]`, one fused multiply-add under
# -ffp-contract=on) as an in-order reduction on arm64: its vectors' products rounded apart, its scalar remainder's
# fused. Whether a loop is vectorised depends on where it is inlined, so the same sum rounded otherwise from one
# caller to the next: a complex128 matrix times a vector, row by row, against each row's own dot product (732 of
# 1680 shapes). Kept scalar, every such sum is fused; in order, the vectors were no faster (300 drawn cases, M1 Pro:
# 1.002). A compiler that does not take the option (GCC, which fuses none of these sums) builds without it.
ordered_reductions_off = ["-mllvm", "-force-ordered-reductions=false"]


class build_ext_with_blas(build_ext):
    """Builds, beside the extension, the numerical_blas library (blas/numerical_blas.cpp): the BLAS and LAPACK
    routines under their Fortran names, for a program linked against those libraries (LD_PRELOAD, FlexiBLAS, the
    system's alternatives). On Linux alone, where a library preloaded or switched in takes the place of the
    system's; numerical.blas_library() gives its path."""

    def build_extensions(self):
        self._compile_sources_in_parallel()
        self.cxx_args = list(compile_args)  # the extension's, and the library's in run()
        if sys.platform != "win32" and self._compiles_with(ordered_reductions_off):
            self.cxx_args += ordered_reductions_off
        for extension in self.extensions:
            extension.extra_compile_args = self.cxx_args
        super().build_extensions()

    def _compile_sources_in_parallel(self):
        """Compiles an extension's sources side by side, one compiler each, where setuptools compiles them one after
        another: the binding is split into units by family so that its build takes several cores."""
        one_by_one = self.compiler.compile

        def compile(sources, *args, **kwargs):
            if len(sources) < 2:
                return one_by_one(sources, *args, **kwargs)
            with ThreadPoolExecutor(max_workers=min(len(sources), os.cpu_count() or 1)) as pool:
                objects = pool.map(lambda source: one_by_one([source], *args, **kwargs), sources)
                return [path for built in objects for path in built]

        self.compiler.compile = compile

    def _compiles_with(self, flags):
        """Whether the extensions' compiler builds a translation unit with `flags`."""
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "probe.cpp")
            with open(source, "w") as out:
                out.write("int main() { return 0; }\n")
            try:
                self.compiler.compile([source], output_dir=directory, extra_postargs=[*compile_args, *flags])
            except CompileError:
                return False
        return True

    def run(self):
        super().run()
        if not sys.platform.startswith("linux"):
            return
        # In a directory of its own, without __init__.py: a .so beside the package's modules would be taken for an
        # extension module (numerical.libnumerical_blas), by an import as by the tools that walk the package.
        package = os.path.join(os.path.dirname(self.get_ext_fullpath("numerical._numerical")), "lib")
        os.makedirs(package, exist_ok=True)
        library = os.path.join(package, "libnumerical_blas.so")
        command = [*self.compiler.compiler_cxx, *self.cxx_args, "-fPIC", "-shared", "-Iinclude",
                   os.path.join("blas", "numerical_blas.cpp"), "-o", library, *link_args,
                   "-Wl,--version-script=" + os.path.join("blas", "numerical_blas.map")]
        self.announce(" ".join(command), level=2)
        subprocess.run(command, check=True)


setup(
    cmdclass={"bdist_wheel": abi3_wheel, "build_ext": build_ext_with_blas, "build_py": build_py_with_headers},
    ext_modules=[
        Extension(
            "numerical._numerical",
            # The binding by family, its helpers in binding.hpp: units a few cores compile side by side.
            sources=[os.path.join("python", "src", name + ".cpp")
                     for name in ("_numerical", "linalg", "eigen", "fft", "elementwise", "products")],
            depends=[os.path.join("python", "src", "binding.hpp")],
            include_dirs=["include", _sciforge_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("Py_LIMITED_API", "0x030A0000")],
            py_limited_api=True,
        )
    ],
)
