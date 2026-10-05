"""Talk to Ollama over its local HTTP API. Localhost only.

This is one of only two places in the project allowed to use the network
(the other is the Update section, step 11). Three safety rules are enforced
here in code, not just by convention:

  1. Loopback only. The URL must point at this machine (localhost,
     127.0.0.1, ::1). Anything else is refused when the backend is created,
     even if console.toml says otherwise.
  2. No proxies. Python normally obeys HTTP_PROXY / HTTPS_PROXY environment
     variables, which could send your data through another machine. We
     switch that off.
  3. No redirects. If the server answers "go to some other address", we
     refuse instead of following it.

We use urllib (built into Python) instead of the popular "requests" package:
one less dependency, and everything that touches the network is in this file.

Ollama endpoints used:
  GET  /api/version  -> {"version": "0.6.2"}
  GET  /api/tags     -> installed models
  POST /api/show     -> details for one model (context length, capabilities)
  POST /api/chat     -> the actual question; "format" = our JSON schema
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from core.backends.base import BackendError, ChatResult, ModelBackend, ModelInfo

# Small calls (listing models) should answer fast. The chat call gets the
# protocol's own timeout instead.
QUICK_TIMEOUT = 10

# Never read more than this from the server in one response.
MAX_RESPONSE_BYTES = 20_000_000


def check_loopback_url(url: str) -> str:
    """Return the URL without a trailing slash, or raise BackendError if it
    doesn't point at this machine."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BackendError(f"backend URL must start with http:// (got {url!r})")
    host = parts.hostname or ""
    if parts.username or parts.password:
        raise BackendError("backend URL must not contain a username or password")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError:
            raise BackendError(
                f"refusing backend address {host!r}: only this machine "
                "(localhost, 127.0.0.1, ::1) is allowed"
            ) from None
    return url.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BackendError(f"model server tried to redirect to {newurl!r}; refused")


def _parse_billions(text: str | None) -> float | None:
    """Turn Ollama's "8.0B" / "137M" / "70B" into a number of billions."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([KMBT])\s*", text or "", re.IGNORECASE)
    if not match:
        return None
    scale = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}[match.group(2).upper()]
    return float(match.group(1)) * scale


class OllamaBackend(ModelBackend):
    kind = "ollama"

    def __init__(self, url: str = "http://localhost:11434"):
        self.url = check_loopback_url(url)
        # ProxyHandler({}) = ignore proxy settings; _NoRedirect = never follow.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    # -- the one function that actually does network I/O ---------------------
    def _request(self, path: str, body: dict | None = None,
                 timeout: float = QUICK_TIMEOUT) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.url + path, data=data, method="GET" if body is None else "POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            detail = e.read(2000).decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise BackendError(f"Ollama returned HTTP {e.code} for {path}: {detail}") from None
        except (socket.timeout, TimeoutError):
            raise BackendError(f"Ollama did not answer {path} within {timeout:.0f}s") from None
        except urllib.error.URLError as e:
            raise BackendError(
                f"can't reach Ollama at {self.url} ({e.reason}). "
                "Is it running? Start it with: ollama serve"
            ) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise BackendError(f"Ollama's answer to {path} was too large")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise BackendError(f"Ollama sent something that isn't JSON for {path}") from None

    # -- the interface ---------------------------------------------------------
    def version(self) -> str:
        return str(self._request("/api/version").get("version", "unknown"))

    def list_models(self) -> list[ModelInfo]:
        models = []
        for entry in self._request("/api/tags").get("models", []):
            details = entry.get("details") or {}
            models.append(ModelInfo(
                name=entry.get("name") or entry.get("model", "?"),
                family=details.get("family", ""),
                parameters_billions=_parse_billions(details.get("parameter_size")),
                size_bytes=int(entry.get("size") or 0),
            ))
        return models

    def model_info(self, name: str) -> ModelInfo:
        data = self._request("/api/show", {"model": name})
        details = data.get("details") or {}
        # The context length key is named after the architecture,
        # e.g. "llama.context_length" or "qwen2.context_length".
        context = None
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                context = value
                break
        return ModelInfo(
            name=name,
            family=details.get("family", ""),
            parameters_billions=_parse_billions(details.get("parameter_size")),
            context_length=context,
            capabilities=list(data.get("capabilities") or []),
        )

    def chat_json(self, model: str, system: str, user: str, schema: dict,
                  context_tokens: int, timeout: float) -> ChatResult:
        started = time.monotonic()
        data = self._request("/api/chat", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,     # Ollama constrains the answer to this JSON shape
            "stream": False,      # one complete answer instead of word-by-word
            "options": {
                "temperature": 0,         # same input -> (nearly) same answer
                "num_ctx": context_tokens,
            },
            # Note: no "tools" key. The model is never offered any tools.
        }, timeout=timeout)
        message = data.get("message") or {}
        return ChatResult(
            text=message.get("content", ""),
            model=data.get("model", model),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            duration_seconds=round(time.monotonic() - started, 2),
        )
