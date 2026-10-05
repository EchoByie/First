"""Run every health check in a sensible order and collect the results."""

from __future__ import annotations

from core.config import resolve_path
from core.discovery import discover
from core.health import HealthReport
from core.health import deps, models, ollama, paths, protocols, refdata


def run_all(config: dict, backend=None, protocols_dir=None) -> HealthReport:
    report = HealthReport()
    discovered = discover(protocols_dir or resolve_path(config, "protocols"))
    results, live_backend = ollama.check(config, backend)
    report.results += results
    report.results += models.check(config, live_backend, discovered.protocols)
    report.results += refdata.check(config, discovered.protocols)
    report.results += protocols.check(discovered)
    report.results += deps.check()
    report.results += paths.check(config)
    return report
