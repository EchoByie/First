"""Health checks: plain code that tells you whether the console is ready.

Each check returns one or more CheckResult lines with a status:
    ok    green  - fine
    warn  amber  - works, but something should be looked at (e.g. old data)
    fail  red    - something a protocol needs is missing or broken

The checks never call the model and never touch the network except asking
the local Ollama server for its version and model list. They are shared by
`aicore health` and the dashboard's Health pane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OK, WARN, FAIL = "ok", "warn", "fail"
_RANK = {OK: 0, WARN: 1, FAIL: 2}


@dataclass
class CheckResult:
    area: str        # e.g. "ollama", "models", "refdata"
    name: str        # what exactly was checked
    status: str      # ok | warn | fail
    message: str     # one line for people
    fix: str = ""    # what to do about it (empty when ok)


@dataclass
class HealthReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def overall(self) -> str:
        return max((r.status for r in self.results), key=_RANK.get, default=OK)

    def by_area(self) -> dict[str, list[CheckResult]]:
        areas: dict[str, list[CheckResult]] = {}
        for r in self.results:
            areas.setdefault(r.area, []).append(r)
        return areas

    def area_status(self, area: str) -> str:
        return max((r.status for r in self.results if r.area == area), key=_RANK.get, default=OK)


def parse_version(text: str) -> tuple[int, ...]:
    """'0.6.2' -> (0, 6, 2); '0.5.0-rc1' -> (0, 5, 0); junk -> ()."""
    parts = []
    for piece in str(text).strip().lstrip("v").split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)
