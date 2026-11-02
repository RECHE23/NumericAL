#!/usr/bin/env python3
"""Run every ```pycon block of the site's pages as a doctest, page by page.

The site's Python examples are shown as interactive sessions; this makes each shown result a
checked one. The blocks of one page run in order in one namespace (a page builds on its earlier
blocks), with whitespace normalized so a long result may wrap across lines.

Usage:
  python3 tools/check_site_examples.py [ROOT]    # docs/site under ROOT (default: this repository)
  python3 tools/check_site_examples.py --self-test
"""
from __future__ import annotations

import doctest
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"^```pycon\n(.*?)^```", re.S | re.M)


def examples(page_text: str) -> str:
    """The page's pycon blocks, joined into one doctest text."""
    return "\n".join(block.rstrip("\n") + "\n" for block in BLOCK.findall(page_text))


def run_page(name: str, page_text: str, report: bool = True) -> tuple[int, int]:
    """(failures, examples run) for one page; with `report`, each difference is printed."""
    text = examples(page_text)
    if not text.strip():
        return 0, 0
    parser = doctest.DocTestParser()
    test = parser.get_doctest(text, {}, name, name, 0)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    result = runner.run(test, out=None if report else (lambda _text: None))
    return result.failed, result.attempted


def check(site: Path) -> int:
    failed = attempted = 0
    pages = sorted(site.rglob("*.md"))
    for page in pages:
        f, a = run_page(str(page.relative_to(site.parent.parent)), page.read_text(encoding="utf-8"))
        failed += f
        attempted += a
    if failed:
        print(f"check_site_examples: {failed} of {attempted} examples differ from what the site shows")
        return 1
    if attempted == 0:
        print("check_site_examples: no pycon example found -- the check would pass on anything")
        return 1
    print(f"check_site_examples: OK -- {attempted} examples across {len(pages)} pages run as shown")
    return 0


def self_test() -> int:
    good = "Text.\n\n```pycon\n>>> x = [1,\n...      2]\n>>> x\n[1, 2]\n```\n\n```pycon\n>>> len(x)\n2\n```\n"
    cases = (
        ("a page whose results hold passes", good, 0),
        ("a wrong result fails", good.replace("[1, 2]\n```", "[1, 3]\n```"), 1),
        ("an example that raises fails", good.replace(">>> len(x)", ">>> len(y)"), 1),
        ("text outside pycon blocks is not run", "```python\n>>> 1/0\n```\n" + good, 0),
    )
    bad = 0
    for label, page, expected in cases:
        failures, _ = run_page("self-test", page, report=False)
        if (failures > 0) != (expected > 0):
            bad += 1
            print(f"check_site_examples --self-test: '{label}' gave {failures} failure(s)")
    if bad:
        return 1
    print(f"check_site_examples --self-test: {len(cases)} cases as expected")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    sys.exit(check((Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT) / "docs" / "site"))
