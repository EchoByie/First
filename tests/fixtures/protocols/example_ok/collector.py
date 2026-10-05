"""Test collector: hands the user's input straight to the model."""


def collect(ctx):
    return {"text": ctx.user_input_text(), "meta": {"source": "user input"},
            "caveats": ["collector caveat"]}
