# Builds the abi3 C++ extension for the NumericAL Python binding. Run from the
# repository root (the sdist/wheel build entry); metadata lives in pyproject.toml.
# NumericAL is header-only (headers in include/) and a leaf — it depends on nothing
# beyond the C++ standard library.
import os
import shutil
import sys

from setuptools import Extension, setup
from setuptools.command.build_py import build_py

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
    compile_args = ["-std=c++20", "-O2", "-fvisibility=hidden", "-pthread"]
    link_args = ["-pthread"]  # parallel.hpp uses std::thread

setup(
    cmdclass={"bdist_wheel": abi3_wheel, "build_py": build_py_with_headers},
    ext_modules=[
        Extension(
            "numerical._numerical",
            sources=["python/src/_numerical.cpp"],
            include_dirs=["include", _sciforge_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("Py_LIMITED_API", "0x030A0000")],
            py_limited_api=True,
        )
    ],
)
