"""The interface every model backend must provide.

Why an interface? Today we talk to Ollama, but the AI core should outlive any
one model or server. Every backend (Ollama now, maybe llama.cpp's server or
another local server later) provides these same few methods. The rest of the
framework only ever calls these methods, so swapping the backend never
touches a protocol.

Note what's NOT here: no tools, no function calling, no internet. A backend
can only list models and turn text into a JSON answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class BackendError(Exception):
    """Something went wrong talking to the model server. The message is
    written for a person to read ("Ollama isn't running..."), not a stack trace."""


@dataclass
class ModelInfo:
    name: str                         # e.g. "llama3.1:8b"
    family: str = ""                  # e.g. "llama"
    parameters_billions: float | None = None   # e.g. 8.0 (model size, rough "smartness")
    context_length: int | None = None          # max tokens the model can read at once
    capabilities: list[str] = field(default_factory=list)  # e.g. ["completion", "vision"]
    size_bytes: int = 0               # size on disk

    @property
    def can_chat(self) -> bool:
        """Embedding-only models can't write answers, so they can't fill a role.
        If the server doesn't report capabilities at all, assume chat works."""
        return not self.capabilities or "completion" in self.capabilities


@dataclass
class ChatResult:
    text: str                  # the model's raw answer (checked later by validation.py)
    model: str
    prompt_tokens: int = 0     # how much it read
    output_tokens: int = 0     # how much it wrote
    duration_seconds: float = 0.0


class ModelBackend(ABC):
    """Every backend implements these four methods."""

    kind = "base"
    url = ""   # where the server is, shown in the console

    @abstractmethod
    def version(self) -> str:
        """Server version, e.g. "0.6.2". Raises BackendError if unreachable."""

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """All installed models (details may be partial; see model_info)."""

    @abstractmethod
    def model_info(self, name: str) -> ModelInfo:
        """Full details for one model, including context_length if known."""

    @abstractmethod
    def chat_json(self, model: str, system: str, user: str, schema: dict,
                  context_tokens: int, timeout: float) -> ChatResult:
        """Send one system + user message, ask for JSON matching `schema`.

        context_tokens: how much the model should be able to read. Ollama
        otherwise uses a small default and silently cuts off long inputs.
        """
