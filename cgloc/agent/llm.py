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
        attempts = 7
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                break
            except OSError:
                # Long enough to ride out a server restart (model reload takes ~1-2 min on a T4).
                if attempt == attempts - 1:
                    raise
                time.sleep(min(10 * (attempt + 1), 60))
        usage = resp.get("usage", {})
        msg = resp["choices"][0]["message"]
        # Gemma 4 chat templates may route output into a separate reasoning channel.
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return Reply(content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), time.time() - t)
