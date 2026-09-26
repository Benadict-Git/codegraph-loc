from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Reply:
    content: str
    prompt_tokens: int
    completion_tokens: int
    secs: float


class OpenAIChat:
    """Minimal client for an OpenAI-compatible /v1/chat/completions endpoint (llama-server, vLLM)."""

    def __init__(self, base_url: str, model: str = "local", max_tokens: int = 1024, temperature: float = 0.0,
                 thinking: bool = False, timeout: int = 900):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.thinking, self.timeout = thinking, timeout

    def __call__(self, messages: list[dict]) -> Reply:
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "chat_template_kwargs": {"enable_thinking": self.thinking}}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t = time.time()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        usage = resp.get("usage", {})
        return Reply(resp["choices"][0]["message"].get("content") or "", usage.get("prompt_tokens", 0),
                     usage.get("completion_tokens", 0), time.time() - t)
