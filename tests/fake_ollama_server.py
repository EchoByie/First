"""A tiny local HTTP server that pretends to be Ollama.

It lets us test the REAL network code in core/backends/ollama.py without
Ollama installed. It only listens on 127.0.0.1 and only while a test runs.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TAGS = {"models": [
    {"name": "llama3.1:8b", "size": 4_900_000_000,
     "details": {"family": "llama", "parameter_size": "8.0B"}},
    {"name": "nomic-embed-text:latest", "size": 270_000_000,
     "details": {"family": "nomic-bert", "parameter_size": "137M"}},
]}
SHOW = {
    "llama3.1:8b": {"details": {"family": "llama", "parameter_size": "8.0B"},
                    "model_info": {"general.architecture": "llama",
                                   "llama.context_length": 131072},
                    "capabilities": ["completion", "tools"]},
    "nomic-embed-text:latest": {"details": {"family": "nomic-bert", "parameter_size": "137M"},
                                "model_info": {"nomic-bert.context_length": 2048},
                                "capabilities": ["embedding"]},
}


class FakeOllama:
    def __init__(self):
        self.requests = []        # (method, path, body) for every request
        self.chat_reply = '{"summary": "ok", "findings": []}'
        self.redirect_to = None   # set to make every request redirect

        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, body):
                fake.requests.append((self.command, self.path, body))
                if fake.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", fake.redirect_to)
                    self.end_headers()
                    return
                if self.path == "/api/version":
                    return self._send(200, {"version": "0.9.9"})
                if self.path == "/api/tags":
                    return self._send(200, TAGS)
                if self.path == "/api/show":
                    if body["model"] not in SHOW:
                        return self._send(404, {"error": f"model '{body['model']}' not found"})
                    return self._send(200, SHOW[body["model"]])
                if self.path == "/api/chat":
                    return self._send(200, {
                        "model": body["model"],
                        "message": {"role": "assistant", "content": fake.chat_reply},
                        "prompt_eval_count": 120, "eval_count": 30,
                    })
                self._send(404, {"error": "not found"})

            def do_GET(self):
                self._handle(None)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self._handle(json.loads(self.rfile.read(length) or b"null"))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)  # port 0 = any free port
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
