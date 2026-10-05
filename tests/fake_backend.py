"""A pretend model backend for tests. No network, no Ollama.

You give it a list of installed models and a list of answers. Each call to
chat_json() returns the next answer and records what was asked, so tests
can check exactly what the framework sent to the model.
"""

from core.backends.base import BackendError, ChatResult, ModelBackend, ModelInfo


class FakeBackend(ModelBackend):
    kind = "fake"
    url = "fake://local"

    def __init__(self, models=None, replies=None, fail=False, responder=None):
        self.models = models if models is not None else [
            ModelInfo("big:70b", "llama", 70.0, 131072, ["completion"]),
            ModelInfo("mid:8b", "llama", 8.0, 131072, ["completion"]),
            ModelInfo("tiny:1b", "llama", 1.0, 8192, ["completion"]),
            ModelInfo("embed:latest", "bert", 0.3, 8192, ["embedding"]),
        ]
        self.replies = list(replies or [])
        self.fail = fail
        self.responder = responder   # optional function(call) -> reply text
        self.calls = []

    def _check(self):
        if self.fail:
            raise BackendError("fake backend is down")

    def version(self):
        self._check()
        return "fake-1.0"

    def list_models(self):
        self._check()
        return list(self.models)

    def model_info(self, name):
        self._check()
        for m in self.models:
            if m.name == name:
                return m
        raise BackendError(f"no model {name}")

    def chat_json(self, model, system, user, schema, context_tokens, timeout):
        self._check()
        call = dict(model=model, system=system, user=user, schema=schema,
                    context_tokens=context_tokens, timeout=timeout)
        self.calls.append(call)
        if self.responder is not None:
            return ChatResult(text=self.responder(call), model=model)
        if not self.replies:
            raise BackendError("fake backend has no more replies")
        return ChatResult(text=self.replies.pop(0), model=model)
