"""Is the reference data there, and fresh enough for each protocol?"""

from __future__ import annotations

from core.config import resolve_path
from core.health import FAIL, OK, WARN, CheckResult
from core.refdata.db import RefDataError, RefDB

DEFAULT_MAX_AGE_DAYS = 90
# Registries the "oui" dataset must have to be useful at all.
ESSENTIAL = {"oui": {"MA-L"}}
FIX_IMPORT = "open the Update section (aicore update) or: aicore refdata import <files>"


def check(config: dict, protocols: dict) -> list[CheckResult]:
    needed: dict[str, int] = {}          # dataset -> strictest max age
    for manifest in protocols.values():
        for dataset in manifest.required_refdata:
            age = manifest.max_refdata_age_days or DEFAULT_MAX_AGE_DAYS
            needed[dataset] = min(age, needed.get(dataset, age))
    path = resolve_path(config, "reference_db")
    try:
        db = RefDB(path)
    except RefDataError as e:
        status = FAIL if needed else WARN
        return [CheckResult("refdata", "database", status, str(e), FIX_IMPORT)]

    results = []
    with db:
        sources = db.sources()
        results.append(CheckResult("refdata", "database", OK,
                                   f"{path.name}: {sum(s.rows for s in sources):,} entries"))
        for dataset, max_age in sorted(needed.items()):
            mine = [s for s in sources if s.dataset == dataset]
            missing = ESSENTIAL.get(dataset, set()) - {s.registry for s in mine}
            if not mine or missing:
                results.append(CheckResult("refdata", dataset, FAIL,
                                           f"missing: {', '.join(sorted(missing)) or dataset}",
                                           FIX_IMPORT))
                continue
            oldest = max(mine, key=lambda s: s.age_days())
            age = oldest.age_days()
            text = (f"{len(mine)} registr{'y' if len(mine) == 1 else 'ies'}, "
                    f"{sum(s.rows for s in mine):,} entries, oldest {age:.0f} days old")
            if age > max_age:
                results.append(CheckResult("refdata", dataset, WARN,
                                           text + f" (limit {max_age} days)", FIX_IMPORT))
            else:
                results.append(CheckResult("refdata", dataset, OK, text))
    return results
