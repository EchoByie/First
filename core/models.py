"""Decide which installed model plays each role a protocol asks for.

Protocols say what they NEED ("an analyst that can read 8192 tokens").
console.toml says what you WANT ("analyst = auto" or a model name).
This module combines the two with what is actually INSTALLED.

"auto" rules:
  - only models that can chat (not embedding-only models)
  - only models whose context window is big enough (unknown size = allowed,
    with a warning)
  - analyst and any other role: pick the LARGEST model (usually the most capable)
  - worker: pick the SMALLEST model (it handles many chunks, so speed matters)

When you install a better model, "auto" starts using it without any edits.
That is what keeps the console from being tied to one model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.backends.base import BackendError, ModelBackend, ModelInfo
from core.manifest import RoleSpec

# Roles that should prefer a small, fast model under "auto".
PREFER_SMALL = {"worker"}


@dataclass
class RoleChoice:
    role: str
    model: str | None             # None = no suitable model
    context_tokens: int           # what to ask the backend for (num_ctx)
    reason: str                   # one line explaining the choice
    warnings: list[str] = field(default_factory=list)
    model_context: int | None = None   # the model's own maximum, if known

    @property
    def ok(self) -> bool:
        return self.model is not None


def gather_models(backend: ModelBackend) -> list[ModelInfo]:
    """List installed models with full details (context size, capabilities).

    If one model's details can't be fetched we keep the basic listing for
    it rather than failing the whole call.
    """
    detailed = []
    for basic in backend.list_models():
        try:
            info = backend.model_info(basic.name)
        except BackendError:
            info = basic
        # /api/show may omit fields that /api/tags had; fill them back in.
        info.parameters_billions = info.parameters_billions or basic.parameters_billions
        info.family = info.family or basic.family
        info.size_bytes = info.size_bytes or basic.size_bytes
        detailed.append(info)
    return detailed


def _find_installed(name: str, models: list[ModelInfo]) -> ModelInfo | None:
    """Match a configured name to an installed model.
    Ollama treats "llama3.1" as "llama3.1:latest", so we do too."""
    wanted = name if ":" in name else name + ":latest"
    for m in models:
        if m.name in (name, wanted):
            return m
    return None


def _size_key(m: ModelInfo) -> float:
    """How "big" a model is: parameter count, else size on disk."""
    if m.parameters_billions is not None:
        return m.parameters_billions
    return m.size_bytes / 1e9  # rough fallback, ~1 GB per billion at 8-bit


def choose_model(spec: RoleSpec, setting: str, models: list[ModelInfo]) -> RoleChoice:
    """Pick the model for one role. `setting` is the console.toml value."""
    context = spec.min_context

    # --- a specific model was named in console.toml ---------------------------
    if setting and setting != "auto":
        m = _find_installed(setting, models)
        if m is None:
            return RoleChoice(spec.name, None, context,
                              f"{setting} is set in console.toml but not installed "
                              f"(install it with: ollama pull {setting})")
        if not m.can_chat:
            return RoleChoice(spec.name, None, context, f"{m.name} can't write answers (embedding-only)")
        if m.context_length is not None and m.context_length < spec.min_context:
            return RoleChoice(spec.name, None, context,
                              f"{m.name} reads at most {m.context_length} tokens; "
                              f"this protocol needs {spec.min_context}")
        warnings = [] if m.context_length else [f"{m.name} doesn't report its context size"]
        return RoleChoice(spec.name, m.name, context, "set in console.toml", warnings,
                          model_context=m.context_length)

    # --- "auto" ----------------------------------------------------------------
    candidates = [m for m in models if m.can_chat
                  and (m.context_length is None or m.context_length >= spec.min_context)]
    if not candidates:
        return RoleChoice(spec.name, None, context,
                          f"no installed model can chat with at least {spec.min_context} tokens of context")
    small_first = spec.name in PREFER_SMALL
    # Tie-break on context length, then name, so the choice is always the same.
    candidates.sort(key=lambda m: (_size_key(m), m.context_length or 0, m.name),
                    reverse=not small_first)
    best = candidates[0]
    reason = f"auto: {'smallest' if small_first else 'largest'} of {len(candidates)} suitable model(s)"
    warnings = [] if best.context_length else [f"{best.name} doesn't report its context size"]
    return RoleChoice(spec.name, best.name, context, reason, warnings,
                      model_context=best.context_length)


def resolve_roles(roles: dict[str, RoleSpec], role_settings: dict[str, str],
                  models: list[ModelInfo]) -> dict[str, RoleChoice]:
    """Choose a model for every role of one protocol."""
    return {
        name: choose_model(spec, role_settings.get(name, "auto"), models)
        for name, spec in roles.items()
    }


def missing_required(choices: dict[str, RoleChoice], roles: dict[str, RoleSpec]) -> list[str]:
    """Names of required roles with no model. Optional roles may be empty
    (e.g. without a worker, big inputs fall back to the analyst)."""
    return [name for name, c in choices.items() if not c.ok and not roles[name].optional]
