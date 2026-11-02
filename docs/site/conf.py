"""Sphinx configuration for NumericAL's site (docs/site): guides written here, the C++ reference read from Doxygen's
XML through Breathe, the Python reference from the binding's docstrings.

Build with `make docs-site` (or `make docs-site-gate`, every warning fatal). The toolchain is pinned in
docs/requirements.txt and installed into build/.venv-docs by `make docs-venv`, never into the system interpreter.
"""

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # docs/site
_ROOT = _HERE.parent.parent  # repository root

# The Python reference imports the built extension from python/ (make python).
sys.path.insert(0, str(_ROOT / "python"))

project = "NumericAL"
copyright = "2026, René Chenard"
author = "René Chenard"

# Derived from pyproject.toml, the version's single source (make version-check).
_version = re.search(r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"',
                     (_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
if _version is None:
    raise RuntimeError('docs/site/conf.py: no top-level version = "X.Y.Z" in pyproject.toml')
version = release = _version.group(1)

extensions = [
    "myst_parser",
    "breathe",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"

root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

breathe_projects = {"numerical": str(_ROOT / "build" / "doc" / "xml")}
breathe_default_project = "numerical"
breathe_default_members = ("members",)

html_theme = "pydata_sphinx_theme"
html_title = f"NumericAL {version}"
html_static_path = ["_static"]
html_css_files = ["numerical.css"]
html_show_sourcelink = False
html_theme_options = {
    "github_url": "https://github.com/RECHE23/NumericAL",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "external_links": [
        {"name": "API (Doxygen)", "url": "api/index.html"},
        {"name": "REAL", "url": "https://reche23.github.io/real-regex/"},
    ],
}
