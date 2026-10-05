"""Are Python and the packages we depend on recent enough?

Minimum versions are read from our own pyproject.toml, so there is one
place to change them.
"""

from __future__ import annotations

import re
import sys
import tomllib
from importlib import metadata

from core.config import PROJECT_ROOT
from core.health import FAIL, OK, CheckResult, parse_version

MIN_PYTHON = (3, 11)


def _minimums() -> dict[str, tuple[int, ...]]:
    try:
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            deps = tomllib.load(f)["project"]["dependencies"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return {}
    wanted = {}
    for dep in deps:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*>=\s*([\d.]+)", dep)
        if match:
            wanted[match.group(1).lower()] = parse_version(match.group(2))
    return wanted


def check() -> list[CheckResult]:
    here = sys.version_info[:3]
    results = [CheckResult(
        "deps", "python", OK if here >= MIN_PYTHON else FAIL,
        "Python " + ".".join(map(str, here)),
        "" if here >= MIN_PYTHON else "install Python 3.11 or newer")]
    for package, minimum in sorted(_minimums().items()):
        want = ".".join(map(str, minimum))
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            results.append(CheckResult("deps", package, FAIL, "not installed",
                                       'run: pip install -e ".[dev]"'))
            continue
        if parse_version(version) < minimum:
            results.append(CheckResult("deps", package, FAIL, f"{version} (needs {want}+)",
                                       f"run: pip install -U {package}"))
        else:
            results.append(CheckResult("deps", package, OK, version))
    return results
