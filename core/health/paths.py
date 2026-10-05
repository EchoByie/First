"""Can the console write its reports and audit log?"""

from __future__ import annotations

import os

from core.config import resolve_path
from core.health import FAIL, OK, CheckResult


def _writable(folder) -> bool:
    folder.mkdir(parents=True, exist_ok=True)
    return os.access(folder, os.W_OK)


def check(config: dict) -> list[CheckResult]:
    results = []
    for key, folder in (("reports", resolve_path(config, "reports")),
                        ("audit log", resolve_path(config, "audit_log").parent)):
        try:
            ok = _writable(folder)
        except OSError:
            ok = False
        results.append(CheckResult("paths", key, OK if ok else FAIL,
                                   f"{folder} is {'writable' if ok else 'NOT writable'}",
                                   "" if ok else "fix the folder's permissions or [paths] in console.toml"))
    return results
