"""Do all protocols load, are their schemas valid, and can their collectors
reach what they need on THIS platform? (Checked without running them.)"""

from __future__ import annotations

import os
import shutil

from core.discovery import current_platform
from core.health import FAIL, OK, WARN, CheckResult
from core.validation import SchemaFileError, build_answer_schema, load_schema


def check(discovered) -> list[CheckResult]:
    here = current_platform()
    results = []
    for error in discovered.broken:
        results.append(CheckResult("protocols", error.folder.name, FAIL,
                                   "; ".join(error.problems[:3]),
                                   "fix the manifest (see `aicore list`)"))
    for manifest in discovered.protocols.values():
        name = manifest.name
        try:
            build_answer_schema(load_schema(manifest.files["output_schema"]), manifest.finding_kinds)
        except SchemaFileError as e:
            results.append(CheckResult("protocols", name, FAIL, f"bad output schema: {e}"))
            continue
        if not manifest.supports(here):
            results.append(CheckResult("protocols", name, WARN, f"not available on {here}"))
            continue
        commands, files = manifest.policy_for(here)
        missing = [c[0] for c in commands if shutil.which(c[0]) is None]
        unreadable = [f for f in files if not os.access(f, os.R_OK)]
        if missing or unreadable:
            what = ", ".join([f"program {m}" for m in missing] + [f"file {f}" for f in unreadable])
            results.append(CheckResult("protocols", name, FAIL, f"can't reach: {what}"))
        else:
            needs = len(commands) + len(files)
            results.append(CheckResult("protocols", name, OK,
                                       f"ready ({needs} allow-listed source(s) available)"
                                       if needs else "ready"))
    if not results:
        results.append(CheckResult("protocols", "discovery", WARN, "no protocols found"))
    return results
