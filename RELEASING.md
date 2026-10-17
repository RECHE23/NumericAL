# Releasing NumericAL

`make release` computes the next calendar version `YYYY.M.PATCH` — the patch
resets each month, the first release of a month is `.0` (PEP 440 drops leading
zeros, so `2026.6.1`, never `2026.06.001`) — bumps it in `pyproject.toml` and
`python/numerical/__init__.py`, then commits, tags and pushes from a clean `main`.
Pushing the tag drives `release.yml`, which checks the tag matches the version,
builds the abi3 wheels (Linux x86-64/aarch64, macOS universal, Windows) and the
sdist, and publishes to PyPI via Trusted Publishing (OIDC, no stored secret).

One-time PyPI setup (before the first release): create a
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for the project
`numeric-al` — owner `RECHE23`, repository `NumericAL`, workflow `release.yml`,
environment `pypi` — and create the matching GitHub Environment named `pypi`.
Until that exists the build/sdist jobs still run, but the publish step fails;
the pushed tag remains a valid versioned snapshot regardless.

## Versions and breaking changes

Versions are calendar-based, `YYYY.M.PATCH`. A change that can break a user's code or results it depends on — a
removed or renamed function, a changed signature or default, a C++ enumerator's value, a different answer for the
same input beyond its last bits — is listed under a **Breaking** heading of its release's CHANGELOG entry, and is
held for the first release of a month (`YYYY.M.0`), so that the patch releases within a month only add and fix.
Changes in the last bits of a floating-point result, from a faster kernel summing in another order, are not
breaking: each is noted under **Changed**.
