"""Model backends. make_backend() picks one based on config/console.toml."""

from __future__ import annotations

from core.backends.base import BackendError, ModelBackend


def make_backend(config: dict) -> ModelBackend:
    kind = config["backend"]["kind"]
    if kind == "ollama":
        from core.backends.ollama import OllamaBackend
        return OllamaBackend(config["backend"]["url"])
    raise BackendError(f"unknown backend kind {kind!r} in console.toml (supported: ollama)")
