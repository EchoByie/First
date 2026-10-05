"""Shared test helpers. pytest loads this file automatically."""

import sys
from pathlib import Path

import pytest

# Let tests do `from fake_backend import FakeBackend`.
sys.path.insert(0, str(Path(__file__).parent))

from fake_ollama_server import FakeOllama  # noqa: E402


@pytest.fixture
def fake_ollama():
    with FakeOllama() as server:
        yield server
