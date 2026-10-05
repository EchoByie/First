"""Does every protocol role have a model?"""

from __future__ import annotations

from core.backends import BackendError
from core.health import FAIL, OK, WARN, CheckResult
from core.models import gather_models, resolve_roles


def check(config: dict, backend, protocols: dict) -> list[CheckResult]:
    if backend is None:
        return [CheckResult("models", "installed", WARN, "skipped: model server not reachable")]
    try:
        models = gather_models(backend)
    except BackendError as e:
        return [CheckResult("models", "installed", FAIL, str(e))]
    chat = [m for m in models if m.can_chat]
    if not chat:
        return [CheckResult("models", "installed", FAIL, "no chat models installed",
                            "install one, e.g.: ollama pull llama3.1:8b")]
    results = [CheckResult("models", "installed", OK,
                           f"{len(chat)} chat model(s): " + ", ".join(m.name for m in chat[:6])
                           + (" ..." if len(chat) > 6 else ""))]
    for manifest in protocols.values():
        for role, choice in resolve_roles(manifest.roles, config["roles"], models).items():
            name = f"{manifest.name} / {role}"
            if choice.ok:
                status = WARN if choice.warnings else OK
                results.append(CheckResult("models", name, status,
                                           f"{choice.model} ({choice.reason})"
                                           + ("; " + "; ".join(choice.warnings) if choice.warnings else "")))
            elif manifest.roles[role].optional:
                results.append(CheckResult("models", name, WARN, f"none: {choice.reason}",
                                           "optional: the analyst model will be used instead"))
            else:
                results.append(CheckResult("models", name, FAIL, f"none: {choice.reason}",
                                           "install a suitable model or change [roles] in console.toml"))
    return results
