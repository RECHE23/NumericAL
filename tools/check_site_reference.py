#!/usr/bin/env python3
"""Check that the site's Python reference documents every public name, and no name that does not exist.

Every name of ``numerical.__all__``, every public name of ``numerical.fft``, and every name of ``numerical.linalg``
that is not the root's function of that name must appear once as an ``auto*`` directive of
docs/site/reference/python.rst, so that a function added to the package cannot go unlisted.

Usage:
  PYTHONPATH=python python3 tools/check_site_reference.py
  python3 tools/check_site_reference.py --self-test
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRECTIVE = re.compile(r"^\.\. auto(?:function|class|exception):: numerical\.(\S+)$", re.M)


def problems(page_text: str, public: list[str]) -> list[str]:
    """What is wrong with the page against the public names: each missing, unknown or repeated one."""
    documented = DIRECTIVE.findall(page_text)
    counts = collections.Counter(documented)
    known = set(public)
    found = [f"missing: {name}" for name in public if name not in counts]
    found += [f"not a public name: {name}" for name in counts if name not in known]
    found += [f"documented {count} times: {name}" for name, count in counts.items() if count > 1]
    return found


def public_names() -> list[str]:
    import numerical
    import numerical.fft
    import numerical.linalg
    fft = [f"fft.{name}" for name in dir(numerical.fft)
           if not name.startswith("_") and name not in ("Tensor", "asarray") and callable(getattr(numerical.fft, name))]
    linalg = [f"linalg.{name}" for name in numerical.linalg.__all__
              if getattr(numerical.linalg, name) is not getattr(numerical, name, None)]  # the root's documented there
    return list(numerical.__all__) + fft + linalg


def check() -> int:
    page = (ROOT / "docs" / "site" / "reference" / "python.rst").read_text(encoding="utf-8")
    names = public_names()
    found = problems(page, names)
    for line in found:
        print(f"check_site_reference: {line}")
    if found:
        return 1
    print(f"check_site_reference: OK -- {len(names)} public names, each documented once")
    return 0


def self_test() -> int:
    page = ".. autofunction:: numerical.a\n.. autoclass:: numerical.B\n.. autofunction:: numerical.fft.c\n"
    cases = (
        ("a page listing every name passes", page, ["a", "B", "fft.c"], 0),
        ("a name left out fails", page, ["a", "B", "fft.c", "d"], 1),
        ("a name that is not public fails", page, ["a", "fft.c"], 1),
        ("a name listed twice fails", page + ".. autofunction:: numerical.a\n", ["a", "B", "fft.c"], 1),
    )
    bad = 0
    for label, text, names, expected in cases:
        if bool(problems(text, names)) != bool(expected):
            bad += 1
            print(f"check_site_reference --self-test: '{label}' did not hold")
    if bad:
        return 1
    print(f"check_site_reference --self-test: {len(cases)} cases as expected")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if sys.argv[1:] == ["--self-test"] else check())
