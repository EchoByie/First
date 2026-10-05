"""Step 4 tests: choosing a model for each role."""

from pathlib import Path

from fake_backend import FakeBackend

from core.backends.base import BackendError, ModelInfo
from core.cli import cmd_models
from core.manifest import RoleSpec
from core.models import choose_model, gather_models, missing_required, resolve_roles

FIXTURES_PROTOCOLS = Path(__file__).parent / "fixtures" / "protocols"

MODELS = FakeBackend().models   # big:70b, mid:8b, tiny:1b (8k context), embed (can't chat)
ANALYST = RoleSpec("analyst", min_context=8192)
WORKER = RoleSpec("worker", min_context=4096, optional=True)


def test_auto_analyst_picks_largest():
    assert choose_model(ANALYST, "auto", MODELS).model == "big:70b"


def test_auto_worker_picks_smallest_that_can_chat():
    choice = choose_model(WORKER, "auto", MODELS)
    assert choice.model == "tiny:1b"          # not the embedding model, even though smaller
    assert "smallest" in choice.reason


def test_auto_respects_context_requirement():
    needs_big_context = RoleSpec("worker", min_context=32768)
    assert choose_model(needs_big_context, "auto", MODELS).model == "mid:8b"


def test_auto_with_nothing_suitable():
    choice = choose_model(RoleSpec("analyst", min_context=10**6), "auto", MODELS)
    assert not choice.ok and "no installed model" in choice.reason


def test_auto_picks_up_newly_installed_model():
    newer = MODELS + [ModelInfo("future:400b", "next", 400.0, 1_000_000, ["completion"])]
    assert choose_model(ANALYST, "auto", newer).model == "future:400b"


def test_named_model_is_used():
    choice = choose_model(ANALYST, "mid:8b", MODELS)
    assert choice.model == "mid:8b" and choice.reason == "set in console.toml"


def test_named_model_without_tag_means_latest():
    models = [ModelInfo("phi4:latest", "phi", 14.0, 16384, ["completion"])]
    assert choose_model(ANALYST, "phi4", models).model == "phi4:latest"


def test_named_model_problems():
    assert "ollama pull" in choose_model(ANALYST, "missing:7b", MODELS).reason
    assert "embedding" in choose_model(ANALYST, "embed:latest", MODELS).reason
    small_ctx = choose_model(RoleSpec("analyst", min_context=16384), "tiny:1b", MODELS)
    assert not small_ctx.ok and "8192" in small_ctx.reason


def test_unknown_context_size_is_allowed_with_warning():
    models = [ModelInfo("mystery:7b", "", 7.0, None, [])]
    choice = choose_model(ANALYST, "auto", models)
    assert choice.ok and choice.warnings


def test_context_tokens_requested_matches_role():
    assert choose_model(ANALYST, "auto", MODELS).context_tokens == 8192


def test_resolve_roles_and_missing_required():
    roles = {"analyst": RoleSpec("analyst", min_context=10**6), "worker": WORKER}
    choices = resolve_roles(roles, {"analyst": "auto"}, MODELS)
    assert choices["worker"].model == "tiny:1b"          # missing setting -> auto
    assert missing_required(choices, roles) == ["analyst"]


def test_gather_models_survives_one_broken_model():
    class Flaky(FakeBackend):
        def model_info(self, name):
            if name == "mid:8b":
                raise BackendError("boom")
            return super().model_info(name)
    names = [m.name for m in gather_models(Flaky())]
    assert "mid:8b" in names and len(names) == 4


def test_cli_models_shows_models_and_roles(capsys):
    assert cmd_models(FIXTURES_PROTOCOLS, backend=FakeBackend()) == 0
    out = capsys.readouterr().out
    assert "big:70b" in out and "example_ok" in out and "tiny:1b" in out
    assert "console.toml [roles]" in out


def test_cli_models_when_backend_down(capsys):
    assert cmd_models(FIXTURES_PROTOCOLS, backend=FakeBackend(fail=True)) == 1
    assert "down" in capsys.readouterr().out
