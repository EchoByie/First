"""Step 4 tests: the Ollama backend, tested against a fake local Ollama server."""

import socket

import pytest

from core.backends import make_backend
from core.backends.base import BackendError
from core.backends.ollama import OllamaBackend, _parse_billions, check_loopback_url


# --- loopback-only rule -------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434",
    "http://127.0.0.5:11434/", "https://localhost:11434",
])
def test_loopback_urls_are_allowed(url):
    assert check_loopback_url(url) == url.rstrip("/")


@pytest.mark.parametrize("url", [
    "http://192.168.1.50:11434",            # another machine on the LAN
    "http://example.com:11434",             # the internet
    "http://localhost.evil.com:11434",      # looks like localhost, isn't
    "http://127.0.0.1.nip.io:11434",        # DNS trick
    "http://0.0.0.0:11434",                 # "all interfaces", not loopback
    "http://user:pw@localhost:11434",       # credentials in URL
    "ftp://localhost:11434",
    "localhost:11434",                      # no scheme
])
def test_other_urls_are_refused(url):
    with pytest.raises(BackendError):
        OllamaBackend(url)


def test_factory_builds_ollama_and_rejects_unknown():
    config = {"backend": {"kind": "ollama", "url": "http://localhost:11434"}}
    assert isinstance(make_backend(config), OllamaBackend)
    with pytest.raises(BackendError):
        make_backend({"backend": {"kind": "cloud-ai", "url": "x"}})
    with pytest.raises(BackendError):
        make_backend({"backend": {"kind": "ollama", "url": "http://10.0.0.9:11434"}})


# --- talking to the (fake) server ---------------------------------------------

def test_version_and_models(fake_ollama):
    backend = OllamaBackend(fake_ollama.url)
    assert backend.version() == "0.9.9"
    names = [m.name for m in backend.list_models()]
    assert names == ["llama3.1:8b", "nomic-embed-text:latest"]


def test_model_info_reads_context_and_capabilities(fake_ollama):
    backend = OllamaBackend(fake_ollama.url)
    llama = backend.model_info("llama3.1:8b")
    assert llama.context_length == 131072
    assert llama.parameters_billions == 8.0
    assert llama.can_chat
    embed = backend.model_info("nomic-embed-text:latest")
    assert not embed.can_chat


def test_unknown_model_gives_readable_error(fake_ollama):
    with pytest.raises(BackendError, match="not found"):
        OllamaBackend(fake_ollama.url).model_info("nope:1b")


def test_chat_sends_schema_and_no_tools(fake_ollama):
    backend = OllamaBackend(fake_ollama.url)
    schema = {"type": "object"}
    result = backend.chat_json("llama3.1:8b", "SYSTEM", "USER", schema,
                               context_tokens=8192, timeout=5)
    assert result.text == '{"summary": "ok", "findings": []}'
    assert (result.prompt_tokens, result.output_tokens) == (120, 30)

    method, path, body = fake_ollama.requests[-1]
    assert (method, path) == ("POST", "/api/chat")
    assert body["format"] == schema
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "tools" not in body          # the model is never offered tools


def test_proxy_settings_are_ignored(fake_ollama, monkeypatch):
    # If proxies were obeyed, this would try to reach 10.255.255.1 and fail.
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://10.255.255.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert OllamaBackend(fake_ollama.url).version() == "0.9.9"


def test_redirects_are_refused(fake_ollama):
    fake_ollama.redirect_to = "http://example.com/steal"
    with pytest.raises(BackendError, match="redirect"):
        OllamaBackend(fake_ollama.url).version()


def test_server_not_running_gives_friendly_error():
    # Find a free port, then close it, so nothing is listening there.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(BackendError, match="ollama serve"):
        OllamaBackend(f"http://127.0.0.1:{port}").version()


@pytest.mark.parametrize("text, value", [
    ("8.0B", 8.0), ("70B", 70.0), ("137M", 0.137), ("1.5T", 1500.0), ("", None), ("big", None),
])
def test_parameter_sizes(text, value):
    result = _parse_billions(text)
    assert result == pytest.approx(value) if value is not None else result is None
